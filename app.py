import streamlit as st
import pandas as pd
import json
import faiss
from sentence_transformers import SentenceTransformer
import os
import datetime
import requests
from sklearn.metrics.pairwise import cosine_similarity

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="F1 RAG Summarizer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏎️"
)

# --- Custom CSS  ---
st.markdown(
    """
    <style>
    body { background-color: #1a1a1a; color: #ffffff; }
    .stApp { background-color: #1a1a1a; color: #ffffff; }
    .st-emotion-cache-1ldk3x6 { background-color: #2c2c2c; color: #ffffff; } /* Sidebar */
    .st-emotion-cache-1ldk3x6 .st-emotion-cache-1kyx2k0 { color: #e10600; font-weight: bold; } /* Sidebar header */
    
    h1, h2, h3, h4, h5, h6 { color: #e10600; font-family: 'Inter', sans-serif; font-weight: 900; }
    h1 { text-align: center; font-size: 3em; text-shadow: 2px 2px 4px #000000; }
    h2 { font-size: 2em; }

    .stButton > button {
        background-color: #e10600; color: white; border-radius: 8px; border: 2px solid #e10600;
        font-weight: bold; padding: 10px 20px; transition: background-color 0.3s, color 0.3s, border-color 0.3s;
        box-shadow: 3px 3px 6px rgba(0, 0, 0, 0.4);
    }
    .stButton > button:hover { background-color: #ff3333; border-color: #ff3333; color: white; }

    .stTextInput > div > div > input,
    .stSelectbox > div > div > div > div > div,
    .stTextArea > div > div > textarea {
        background-color: #333333; color: #ffffff; border-radius: 5px; border: 1px solid #555555;
        padding: 8px 12px;
    }
    .stTextInput > label, .stSelectbox > label, .stTextArea > label { color: #cccccc; font-weight: bold; }
    
    .st-emotion-cache-1xw8zd0 {
        background-color: #2c2c2c; border-left: 5px solid #e10600; border-radius: 8px;
        padding: 15px; margin-top: 15px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .st-emotion-cache-1xw8zd0 p { color: #ffffff; }

    .st-emotion-cache-lnq4tk {
        background-color: #2c2c2c; border-radius: 8px; border: 1px solid #555555;
        margin-top: 15px; margin-bottom: 15px; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    .st-emotion-cache-lnq4tk > div > div { color: #cccccc; font-weight: bold; padding: 10px; }
    .st-emotion-cache-lnq4tk p { color: #ffffff; }

    hr { border-top: 2px solid #e10600; margin-top: 30px; margin-bottom: 30px; }
    .st-emotion-cache-jeb36u {
        background-color: #2c2c2c; border-radius: 8px; padding: 15px; margin: 5px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Konfigurasi Path File ---
FAISS_INDEX_PATH = 'data/faiss_index.bin'
CHUNKS_PATH = 'data/chunks.json'
CSV_DATA_DIR = 'data/raw_csv'

os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
os.makedirs(CSV_DATA_DIR, exist_ok=True)

# --- Konfigurasi API Google Gemini ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") 
GEMINI_MODEL_NAME = "gemini-1.5-flash" 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# --- Pemuatan Model & Basis Pengetahuan ---
@st.cache_resource
def load_rag_components():
    """Memuat model embedding, FAISS index, dan chunks ke cache Streamlit."""
    with st.spinner("Memuat komponen RAG..."):
        try:
            embedding_model = SentenceTransformer('all-mpnet-base-v2')
            if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
                st.error("File basis pengetahuan tidak ditemukan. Mohon jalankan 'build_knowledge_base.py'.")
                st.stop()
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNKS_PATH, "r") as f:
                f1_chunks_raw = json.load(f)
            f1_chunk_texts = [item['text'] for item in f1_chunks_raw]
        except Exception as e:
            st.error(f"Gagal memuat komponen RAG. Error: {e}")
            st.stop()
    return embedding_model, index, f1_chunks_raw, f1_chunk_texts

embedding_model, faiss_index, f1_chunks_raw, f1_chunk_texts = load_rag_components()

# --- Pemuatan Data Balapan untuk Dropdown ---
@st.cache_data
def load_race_data_for_dropdown():
    """Memuat data balapan dari races.csv, difilter berdasarkan tahun yang relevan."""
    try:
        df_races = pd.read_csv(os.path.join(CSV_DATA_DIR, 'races.csv'))
        df_races['year'] = df_races['year'].astype(int)
        
        # Daftar tahun yang ada di basis pengetahuan (sesuaikan dengan build_knowledge_base.py)
        years_to_display_in_app = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017] 
        df_races_filtered = df_races[df_races['year'].isin(years_to_display_in_app)].copy()
        
        years = sorted(df_races_filtered['year'].unique(), reverse=True)
        return df_races_filtered, years
    except Exception as e:
        st.error(f"Gagal memuat data balapan untuk dropdown: {e}")
        return pd.DataFrame(), []

df_races_global, year_options = load_race_data_for_dropdown()

# --- Fungsi Retrieval ---
def retrieve_relevant_chunks(query, embedding_model, faiss_index, all_chunk_texts, top_k=70):
    """Mencari chunks paling relevan dari FAISS index."""
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
    distances, indices = faiss_index.search(query_embedding, top_k)
    return indices[0]

# --- Fungsi Generasi dengan Google Gemini API ---
def generate_summary_with_gemini(query, relevant_chunks_final_text, api_url, api_key):
    """Menghasilkan ringkasan menggunakan Google Gemini API."""
    if not api_key: return "ERROR: Google Gemini API Key belum diatur."
    if not relevant_chunks_final_text: return "Tidak ada informasi relevan yang cukup."

    document_content = "\n".join([f"- {chunk_text}" for chunk_text in relevant_chunks_final_text])
    
    # Prompt untuk Gemini API
    prompt_text = f"""
Anda adalah komentator dan analis Formula 1 yang ahli dan ringkas.
Tugas Anda adalah membuat ringkasan komprehensif, faktual, akurat, dan mudah dipahami penggemar F1.
**Gunakan HANYA informasi yang disediakan.** Jangan mengarang atau menambahkan informasi yang tidak ada. Jika informasi tertentu tidak tersedia, abaikan saja.

Berikut adalah kumpulan informasi faktual dari balapan Formula 1:
---
{document_content}
---

Buat ringkasan balapan ini.
Fokus utama ringkasan: "{query}".
Sertakan pemenang, posisi finis pembalap kunci (termasuk daftar lengkap jika relevan), lap tercepat, insiden penting (safety car, penalti, kecelakaan), dan strategi tim/pembalap jika relevan.
Pastikan tidak ada pengulangan dan ringkasan disajikan dengan narasi yang baik dan bahasa menarik.
Ringkasan harus setidaknya 3 paragraf jika informasi memadai.
"""
    
    headers = { 'Content-Type': 'application/json' }
    payload = {
        "contents": [{ "parts": [{"text": prompt_text}] }],
        "generationConfig": { "temperature": 0.7, "maxOutputTokens": 1000, "topP": 0.95, "topK": 50 }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if result and 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            st.error(f"Gagal mendapatkan ringkasan dari Gemini API. Respons tidak valid: {json.dumps(result, indent=2)}")
            return "Gagal mendapatkan ringkasan dari Gemini API. Respons tidak valid."
    except requests.exceptions.RequestException as e:
        st.error(f"Terjadi kesalahan saat memanggil Gemini API: {e}. Pastikan API Key Anda benar dan ada koneksi internet.")
        return f"Terjadi kesalahan koneksi/otentikasi dengan Gemini API: {e}"
    except Exception as e:
        st.error(f"Terjadi kesalahan tak terduga: {e}")
        return f"Terjadi kesalahan tak terduga: {e}"

# --- UI Aplikasi Streamlit ---
st.title("🏎️ Formula 1 Race Summarizer")
st.markdown("""
<div style="text-align: center;">
Aplikasi ini menggunakan teknologi Retrieval Augmented Generation (RAG) untuk menyediakan ringkasan balapan Formula 1 yang akurat dan faktual.
</div>
""", unsafe_allow_html=True)

st.header("Tanyakan Sesuatu Tentang Balapan F1! 📊")

col1, col2 = st.columns(2)
selected_year = None
selected_race_name = ""
selected_race_id = None

with col1:
    if year_options:
        selected_year = st.selectbox("Pilih Tahun Balapan: 📅", year_options, help="Tahun balapan yang ingin Anda ketahui.")
    else:
        st.warning("Tidak ada data tahun yang dimuat dari races.csv.")
        selected_year = datetime.datetime.now().year

with col2:
    if not df_races_global.empty and selected_year is not None:
        races_for_year = df_races_global[df_races_global['year'] == selected_year]
        if not races_for_year.empty:
            race_options = {row['name']: row['raceId'] for idx, row in races_for_year.iterrows()}
            selected_race_name = st.selectbox(
                "Pilih Nama Grand Prix: 🏁", 
                list(race_options.keys()), 
                help="Pilih Grand Prix dari daftar yang tersedia untuk tahun ini."
            )
            selected_race_id = race_options.get(selected_race_name)
        else:
            st.warning(f"Tidak ada balapan ditemukan di races.csv untuk tahun {selected_year}.")
    else:
        selected_race_name = st.text_input("Masukkan Nama Grand Prix: 🏁", help="Masukkan nama GP secara manual (contoh: 'Monaco').")

user_query_detail = st.text_area("Apa yang ingin Anda ketahui tentang balapan ini? 📝",
                                 value="hasil balapan, lap tercepat, dan insiden penting",
                                 help="Contoh: 'hasil balapan', 'lap tercepat', 'insiden safety car', 'strategi Red Bull', 'performa Max Verstappen'.")


if st.button("Hasilkan Ringkasan", type="primary"):
    if not GEMINI_API_KEY:
        st.error("Google Gemini API Key belum diatur.")
    elif not selected_race_name.strip() or not user_query_detail.strip():
        st.warning("Mohon pilih Grand Prix atau masukkan nama GP dan detail yang ingin Anda ketahui.")
    elif selected_race_id is None:
        st.error(f"Race ID untuk '{selected_race_name}' tahun {selected_year} tidak ditemukan.")
    else:
        full_query = f"Ringkas balapan {selected_race_name.strip()} {selected_year} dengan fokus pada: {user_query_detail.strip()}"

        with st.spinner("Mencari informasi dan menghasilkan ringkasan..."):
            try:
                # 1. Retrieval Awal: Ambil banyak chunks potensial dari FAISS
                retrieved_chunk_indices = retrieve_relevant_chunks(full_query, embedding_model, faiss_index, f1_chunk_texts, top_k=100)

                # 2. Filtering Berbasis Race ID: Hanya ambil chunks dari balapan yang dipilih
                relevant_chunks_for_this_race = []
                for idx in retrieved_chunk_indices:
                    chunk_item = f1_chunks_raw[idx]
                    if chunk_item['race_id'] == selected_race_id:
                        relevant_chunks_for_this_race.append(chunk_item)

                if not relevant_chunks_for_this_race:
                    st.warning(f"Tidak ada informasi relevan untuk '{selected_race_name.strip()} {selected_year}' terkait '{user_query_detail.strip()}'.")
                else:
                    # 3. Re-ranking: Pilih chunks paling relevan dari yang sudah difilter
                    query_embedding_rerank = embedding_model.encode([full_query])
                    texts_to_rerank = [item['text'] for item in relevant_chunks_for_this_race]
                    embeddings_to_rerank = embedding_model.encode(texts_to_rerank)
                    similarities = cosine_similarity(query_embedding_rerank, embeddings_to_rerank)[0]
                    
                    ranked_chunks_with_score = sorted(
                        zip(similarities, relevant_chunks_for_this_race),
                        key=lambda x: x[0],
                        reverse=True
                    )
                    
                    TOP_K_FINAL_CHUNKS = 25 # Jumlah chunk yang akan dikirim ke Gemini
                    relevant_chunks_final = [chunk_item for score, chunk_item in ranked_chunks_with_score[:TOP_K_FINAL_CHUNKS]]
                    relevant_chunks_final_text_only = [item['text'] for item in relevant_chunks_final]

                    # 4. Generasi: Kirim chunks yang sudah di-rerank ke Gemini API
                    generated_summary = generate_summary_with_gemini(
                        user_query_detail.strip(),
                        relevant_chunks_final_text_only, 
                        GEMINI_API_URL, 
                        GEMINI_API_KEY
                    )

                    st.subheader("🏁 Ringkasan Balapan:")
                    st.info(generated_summary)

                    st.subheader("📚 Informasi yang Digunakan (Chunks):")
                    with st.expander("Klik untuk melihat chunks yang ditemukan", expanded=False):
                        if relevant_chunks_final:
                            for i, chunk_item in enumerate(relevant_chunks_final):
                                st.markdown(f"**Chunk {i+1} (Score: {ranked_chunks_with_score[i][0]:.2f}, Race ID: {chunk_item['race_id']}):** {chunk_item['text']}")
                        else:
                            st.write("Tidak ada chunks yang relevan setelah filtering dan re-ranking.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses permintaan: {e}")
                st.info("Coba sesuaikan pertanyaan atau pastikan data/API Key valid.")

st.markdown("---")
st.markdown("Aplikasi ini menggunakan **Google Gemini API** (`gemini-1.5-flash`).")