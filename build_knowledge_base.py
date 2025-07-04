import pandas as pd
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
import datetime

# --- Konfigurasi File & Direktori ---
FAISS_INDEX_PATH = 'data/faiss_index.bin'
CHUNKS_PATH = 'data/chunks.json'
CSV_DATA_DIR = 'data/raw_csv'

os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
os.makedirs(CSV_DATA_DIR, exist_ok=True)

# --- Memuat Model Embedding ---
print("Memuat model embedding...")
embedding_model = SentenceTransformer('all-mpnet-base-v2')
print("Model embedding berhasil dimuat.")

# --- Pemuatan Semua File CSV dan Preprocessing ---
print("Memuat semua file CSV...")
try:
    df_races = pd.read_csv(os.path.join(CSV_DATA_DIR, 'races.csv'))
    df_results = pd.read_csv(os.path.join(CSV_DATA_DIR, 'results.csv'))
    df_drivers = pd.read_csv(os.path.join(CSV_DATA_DIR, 'drivers.csv'))
    df_constructor_standings = pd.read_csv(os.path.join(CSV_DATA_DIR, 'constructor_standings.csv'))
    df_constructor_results = pd.read_csv(os.path.join(CSV_DATA_DIR, 'constructor_results.csv'))
    df_constructors = pd.read_csv(os.path.join(CSV_DATA_DIR, 'constructors.csv'))
    df_driver_standings = pd.read_csv(os.path.join(CSV_DATA_DIR, 'driver_standings.csv'))
    df_circuits = pd.read_csv(os.path.join(CSV_DATA_DIR, 'circuits.csv'))
    df_seasons = pd.read_csv(os.path.join(CSV_DATA_DIR, 'seasons.csv'))
    df_lap_times = pd.read_csv(os.path.join(CSV_DATA_DIR, 'lap_times.csv'))
    df_qualifying = pd.read_csv(os.path.join(CSV_DATA_DIR, 'qualifying.csv'))
    df_sprint_results = pd.read_csv(os.path.join(CSV_DATA_DIR, 'sprint_results.csv'))
    df_pit_stops = pd.read_csv(os.path.join(CSV_DATA_DIR, 'pit_stops.csv'))
    df_status = pd.read_csv(os.path.join(CSV_DATA_DIR, 'status.csv'))
    print("Semua file CSV berhasil dimuat.")

    # Preprocessing kolom waktu dan durasi
    def safe_to_timedelta(time_str):
        if pd.isna(time_str): return pd.NaT
        try: return pd.to_timedelta(time_str)
        except ValueError:
            try: return pd.to_timedelta('00:' + str(time_str))
            except ValueError: return pd.NaT

    df_lap_times['time_td'] = df_lap_times['time'].apply(safe_to_timedelta)
    df_pit_stops['duration'] = pd.to_numeric(df_pit_stops['duration'], errors='coerce')
    df_pit_stops['milliseconds'] = pd.to_numeric(df_pit_stops['milliseconds'], errors='coerce')
    df_results['milliseconds'] = pd.to_numeric(df_results['milliseconds'], errors='coerce')
    df_results['grid'] = pd.to_numeric(df_results['grid'], errors='coerce').fillna(-1).astype(int)
    df_results['position'] = pd.to_numeric(df_results['position'], errors='coerce').fillna(-1).astype(int)
    df_results['positionOrder'] = pd.to_numeric(df_results['positionOrder'], errors='coerce').fillna(-1).astype(int)

    for col in ['q1', 'q2', 'q3']:
        df_qualifying[f'{col}_td'] = df_qualifying[col].apply(safe_to_timedelta)

    print("Preprocessing kolom CSV berhasil.")

except FileNotFoundError as e:
    print(f"ERROR: Pastikan semua file CSV berada di folder '{CSV_DATA_DIR}'. {e}")
    exit()
except Exception as e:
    print(f"ERROR saat preprocessing CSV: {e}")
    exit()

# --- Fungsi Pembantu (Helpers) ---
def get_driver_info(driver_id):
    driver_info = df_drivers[df_drivers['driverId'] == driver_id]
    if not driver_info.empty: return driver_info.iloc[0]
    return pd.Series({'forename': 'Unknown', 'surname': 'Driver', 'nationality': 'Unknown'})

def get_full_driver_name(driver_id):
    driver_info = get_driver_info(driver_id)
    return f"{driver_info['forename']} {driver_info['surname']}"

def get_constructor_name(constructor_id):
    constructor_info = df_constructors[df_constructors['constructorId'] == constructor_id]
    if not constructor_info.empty: return constructor_info.iloc[0]['name']
    return "Unknown Constructor"

def get_circuit_name(circuit_id):
    circuit_info = df_circuits[df_circuits['circuitId'] == circuit_id]
    if not circuit_info.empty: return circuit_info.iloc[0]['name']
    return "Unknown Circuit"

def get_status_description(status_id):
    status_info = df_status[df_status['statusId'] == status_id]
    if not status_info.empty: return status_info.iloc[0]['status']
    return "Unknown Status"

# --- Fungsi Pembentukan Narasi (Chunks) ---
def create_narrative_chunks_from_dfs(race_id):
    """
    Mengambil data balapan berdasarkan raceId dan mengubahnya menjadi chunks naratif yang detail.
    Setiap chunk dikembalikan sebagai dictionary {'text': ..., 'race_id': ...}.
    """
    chunks_with_metadata = []
    
    race_info = df_races[df_races['raceId'] == race_id]
    if race_info.empty: return []
    race_info = race_info.iloc[0]
    year = int(race_info['year'])
    gp_name = race_info['name']
    circuit_name = get_circuit_name(race_info['circuitId'])
    
    chunks_with_metadata.append({
        'text': f"Balapan Grand Prix {gp_name} tahun {year} adalah putaran ke-{race_info['round']} musim ini, diadakan di sirkuit {circuit_name}. Balapan ini dimulai pada tanggal {race_info['date']}.",
        'race_id': race_id
    })

    race_results = df_results[df_results['raceId'] == race_id].sort_values('positionOrder')
    if not race_results.empty:
        winner_result = race_results.iloc[0]
        winner_info = get_driver_info(winner_result['driverId'])
        winner_name = f"{winner_info['forename']} {winner_info['surname']}"
        winner_nationality = winner_info['nationality']
        winner_constructor = get_constructor_name(winner_result['constructorId'])
        
        chunks_with_metadata.append({
            'text': f"Pemenang balapan {gp_name} {year} adalah {winner_name} asal {winner_nationality} dari tim {winner_constructor}. Dia memulai balapan dari posisi grid {int(winner_result['grid'])} dan berhasil finis di posisi ke-{int(winner_result['position'])} dengan waktu total {winner_result['time']} ({winner_result['milliseconds']} milidetik).",
            'race_id': race_id
        })
        
        # Chunk: Daftar lengkap posisi finis setiap pembalap
        all_finishers_text = []
        for idx, driver_result in race_results.iterrows():
            driver_info = get_driver_info(driver_result['driverId'])
            driver_name = f"{driver_info['forename']} {driver_info['surname']}"
            driver_constructor = get_constructor_name(driver_result['constructorId'])
            final_pos_text = str(driver_result['positionText'])
            status_desc = get_status_description(driver_result['statusId'])
            
            is_dnf = not str(driver_result['positionText']).replace('+', '').isdigit()
            
            if is_dnf:
                all_finishers_text.append(f"- Posisi {final_pos_text} (DNF): {driver_name} ({driver_constructor}) karena '{status_desc}'.")
            else:
                all_finishers_text.append(f"- Posisi {final_pos_text}: {driver_name} ({driver_constructor}). Status: '{status_desc}'.")
        
        if all_finishers_text:
            chunks_with_metadata.append({
                'text': f"Berikut adalah daftar lengkap posisi finis setiap pembalap di Grand Prix {gp_name} {year}:\n" + "\n".join(all_finishers_text),
                'race_id': race_id
            })

        # Detail Pembalap DNF
        non_dnf_status_ids = df_status[df_status['status'].str.contains(r'Finished|\+\d+ Lap', na=False, regex=True)]['statusId'].tolist()
        non_dnf_status_ids.append(1)
        non_dnf_status_ids = list(set(non_dnf_status_ids))

        dnf_drivers = race_results[~race_results['statusId'].isin(non_dnf_status_ids)]

        if not dnf_drivers.empty:
            dnf_list_text = []
            for idx, dnf in dnf_drivers.iterrows():
                driver_name = get_full_driver_name(dnf['driverId'])
                status_desc = get_status_description(dnf['statusId'])
                dnf_list_text.append(f"- {driver_name} ({get_constructor_name(dnf['constructorId'])}) tidak finis karena: '{status_desc}'.")
            
            chunks_with_metadata.append({
                'text': f"Pada balapan {gp_name} {year}, total {len(dnf_drivers)} pembalap tidak menyelesaikan balapan (DNF). Daftar pembalap yang DNF dan alasan mereka adalah:\n" + "\n".join(dnf_list_text),
                'race_id': race_id
            })

    # Lap Tercepat
    race_lap_times = df_lap_times[df_lap_times['raceId'] == race_id].copy()
    if not race_lap_times.empty and 'time_td' in race_lap_times and race_lap_times['time_td'].notna().any():
        fastest_lap_row = race_lap_times.loc[race_lap_times['time_td'].idxmin()]
        fastest_driver_info = get_driver_info(fastest_lap_row['driverId'])
        fastest_driver_name = f"{fastest_driver_info['forename']} {fastest_driver_info['surname']}"
        fastest_lap_time_str = fastest_lap_row['time']
        chunks_with_metadata.append({
            'text': f"Lap tercepat balapan {gp_name} {year} dicatatkan oleh {fastest_driver_name} dengan waktu {fastest_lap_time_str} di Lap {int(fastest_lap_row['lap'])}. Lap tercepat ini juga merupakan lap tercepat di balapan tersebut.",
            'race_id': race_id
        })
        
        if 'winner_result' in locals() and not winner_result.empty:
            winner_driver_id = winner_result['driverId']
            winner_laps = race_lap_times[race_lap_times['driverId'] == winner_driver_id].copy()
            if not winner_laps.empty and 'time_td' in winner_laps and winner_laps['time_td'].notna().any():
                try:
                    avg_lap_time_td = winner_laps['time_td'].mean()
                    total_seconds = avg_lap_time_td.total_seconds()
                    minutes = int(total_seconds // 60)
                    seconds = int(total_seconds % 60)
                    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
                    
                    chunks_with_metadata.append({
                        'text': f"Pemenang {get_full_driver_name(winner_driver_id)} memiliki waktu lap rata-rata sekitar {minutes:02d}:{seconds:02d}.{milliseconds:03d} selama balapan {gp_name} {year}, menunjukkan konsistensi performanya.",
                        'race_id': race_id
                    })
                except Exception as e:
                    print(f"Warning: Gagal menghitung rata-rata lap time untuk {get_full_driver_name(winner_driver_id)} di {gp_name} {year}. Error: {e}")

    # Pit Stops
    race_pit_stops = df_pit_stops[df_pit_stops['raceId'] == race_id].copy()
    if not race_pit_stops.empty and 'duration' in race_pit_stops and race_pit_stops['duration'].notna().any():
        pit_stop_summary_text = []
        pit_stop_summary = race_pit_stops.groupby('driverId').agg(
            stop_count=('stop', 'count'),
            total_duration=('duration', 'sum')
        ).reset_index()

        for idx, row in pit_stop_summary.iterrows():
            driver_name = get_full_driver_name(row['driverId'])
            total_dur_formatted = f"{row['total_duration']:.2f}" if pd.notna(row['total_duration']) else "N/A"
            pit_stop_summary_text.append(f"- Pembalap {driver_name} melakukan {int(row['stop_count'])} kali pit stop dengan total durasi {total_dur_formatted} detik.")
        
        chunks_with_metadata.append({
            'text': f"Ringkasan pit stop di Grand Prix {gp_name} {year}:\n" + "\n".join(pit_stop_summary_text),
            'race_id': race_id
        })

        fastest_pit = race_pit_stops.loc[race_pit_stops['duration'].idxmin()]
        if pd.notna(fastest_pit['duration']):
            fastest_pit_driver = get_full_driver_name(fastest_pit['driverId'])
            chunks_with_metadata.append({
                'text': f"Pit stop tercepat dicatatkan oleh pembalap {fastest_pit_driver} dengan durasi {fastest_pit['duration']:.2f} detik di Lap {int(fastest_pit['lap'])}.",
                'race_id': race_id
            })

    # Hasil Kualifikasi
    race_qualifying = df_qualifying[df_qualifying['raceId'] == race_id].copy()
    if not race_qualifying.empty:
        qual_results_text = []
        for i in range(min(5, len(race_qualifying))):
            qual_result = race_qualifying.iloc[i]
            driver_name = get_full_driver_name(qual_result['driverId'])
            constructor_name = get_constructor_name(qual_result['constructorId'])
            q_times_str = []
            if pd.notna(qual_result['q1']): q_times_str.append(f"Q1: {qual_result['q1']}")
            if pd.notna(qual_result['q2']): q_times_str.append(f"Q2: {qual_result['q2']}")
            if pd.notna(qual_result['q3']): q_times_str.append(f"Q3: {qual_result['q3']}")
            
            q_time_info = f" dengan waktu {', '.join(q_times_str)}" if q_times_str else ""
            qual_results_text.append(f"- Posisi start {int(qual_result['position'])}: {driver_name} ({constructor_name}){q_time_info}.")
        
        if qual_results_text:
            chunks_with_metadata.append({
                'text': f"Hasil kualifikasi teratas untuk Grand Prix {gp_name} {year} adalah:\n" + "\n".join(qual_results_text),
                'race_id': race_id
            })

    # Klasemen Pembalap dan Konstruktor Setelah Balapan Ini
    driver_standings_after_race = df_driver_standings[df_driver_standings['raceId'] == race_id].sort_values('position')
    if not driver_standings_after_race.empty:
        driver_standings_text = []
        for i in range(min(5, len(driver_standings_after_race))):
            standing = driver_standings_after_race.iloc[i]
            driver_name = get_full_driver_name(standing['driverId'])
            driver_standings_text.append(f"- {driver_name} berada di posisi {int(standing['position'])} dengan total {int(standing['points'])} poin, termasuk {int(standing['wins'])} kemenangan musim ini.")
        
        if driver_standings_text:
            chunks_with_metadata.append({
                'text': f"Update klasemen pembalap setelah Grand Prix {gp_name} {year}:\n" + "\n".join(driver_standings_text),
                'race_id': race_id
            })
            
    constructor_standings_after_race = df_constructor_standings[df_constructor_standings['raceId'] == race_id].sort_values('position')
    if not constructor_standings_after_race.empty:
        constructor_standings_text = []
        for i in range(min(5, len(constructor_standings_after_race))):
            standing = constructor_standings_after_race.iloc[i]
            constructor_name = get_constructor_name(standing['constructorId'])
            constructor_standings_text.append(f"- Tim {constructor_name} berada di posisi {int(standing['position'])} dengan total {int(standing['points'])} poin, termasuk {int(standing['wins'])} kemenangan musim ini.")
        
        if constructor_standings_text:
            chunks_with_metadata.append({
                'text': f"Update klasemen konstruktor setelah Grand Prix {gp_name} {year}:\n" + "\n".join(constructor_standings_text),
                'race_id': race_id
            })

    # Sprint Results
    sprint_race_results = df_sprint_results[df_sprint_results['raceId'] == race_id].copy()
    if not sprint_race_results.empty:
        sprint_winner = sprint_race_results.iloc[0]
        sprint_winner_name = get_full_driver_name(sprint_winner['driverId'])
        sprint_winner_constructor = get_constructor_name(sprint_winner['constructorId'])
        chunks_with_metadata.append({
            'text': f"Hasil balapan Sprint di {gp_name} {year}: Pemenang balapan sprint adalah {sprint_winner_name} dari tim {sprint_winner_constructor}. Dia memulai sprint dari posisi grid {int(sprint_winner['grid'])} dan finis di posisi {int(sprint_winner['position'])}.",
            'race_id': race_id
        })
        
        sprint_top_finishers_text = []
        for i in range(1, min(4, len(sprint_race_results))):
            sprint_driver = sprint_race_results.iloc[i]
            driver_name = get_full_driver_name(sprint_driver['driverId'])
            sprint_top_finishers_text.append(f"- Pembalap {driver_name} finis di posisi ke-{int(sprint_driver['position'])} dalam balapan sprint.")
        
        if sprint_top_finishers_text:
            chunks_with_metadata.append({
                'text': f"Beberapa pembalap teratas di balapan Sprint {gp_name} {year}:\n" + "\n".join(sprint_top_finishers_text),
                'race_id': race_id
            })

    return chunks_with_metadata

# --- Pengumpulan Data Balapan untuk Basis Pengetahuan ---
all_f1_chunks = []
all_chunk_texts = []
years_to_process = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017] 
races_to_process_df = df_races[df_races['year'].isin(years_to_process)]

print(f"Mulai membuat chunks naratif untuk {len(races_to_process_df)} balapan dari CSV...")
for idx, race_row in tqdm(races_to_process_df.iterrows(), total=len(races_to_process_df), desc="Memproses Balapan dari CSV"):
    race_id = race_row['raceId']
    chunks_for_race = create_narrative_chunks_from_dfs(race_id)
    if chunks_for_race:
        all_f1_chunks.extend(chunks_for_race)
        all_chunk_texts.extend([chunk['text'] for chunk in chunks_for_race])
print(f"Total {len(all_f1_chunks)} chunks naratif berhasil dibuat.")

# --- Pembuatan Embedding dan Indeks FAISS ---
if not all_f1_chunks:
    print("Tidak ada chunks yang dihasilkan.")
    exit()

print("Membuat embeddings untuk chunks...")
f1_embeddings = embedding_model.encode(all_chunk_texts, convert_to_tensor=True).cpu().numpy()
print("Embeddings berhasil dibuat.")

print("Membangun indeks FAISS...")
d = f1_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(f1_embeddings)
print(f"Indeks FAISS berhasil dibangun dengan {index.ntotal} vektor.")

# --- Menyimpan Basis Pengetahuan ---
print(f"Menyimpan indeks FAISS ke {FAISS_INDEX_PATH}...")
faiss.write_index(index, FAISS_INDEX_PATH)

print(f"Menyimpan chunks teks dan metadata ke {CHUNKS_PATH}...")
with open(CHUNKS_PATH, "w") as f:
    json.dump(all_f1_chunks, f, indent=2)

print("Basis pengetahuan berhasil dibuat dan disimpan!")