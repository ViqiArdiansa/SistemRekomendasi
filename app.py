import io
import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Sistem Rekomendasi Buku")
st.caption(
    "Sistem rekomendasi buku menggunakan pendekatan Hybrid "
    "Collaborative Filtering dan Content-Based Filtering."
)

st.markdown("""
Sistem menggabungkan dua pendekatan:
- **Collaborative Filtering (CF):** memanfaatkan pola rating dari pengguna lain yang memiliki preferensi serupa.
- **Content-Based Filtering (CBF):** memanfaatkan kemiripan karakteristik buku berdasarkan judul, penulis, dan subjek.
""")

with st.sidebar:
    st.header("Pengaturan Sistem")
    st.write("Unggah dataset Book-Crossing yang digunakan pada proses pemodelan.")
    ratings_file = st.file_uploader(
        "Dataset rating (BX-Book-Ratings.csv)",
        type=["csv"],
        key="ratings"
    )
    books_file = st.file_uploader(
        "Dataset buku (BX-Books.csv)",
        type=["csv"],
        key="books"
    )
    top_n = st.selectbox("Jumlah rekomendasi", [5, 10, 15, 20], index=0)
    st.divider()
    st.caption(
        "Parameter model: CF 40%, CBF 60%, lambda penalty 0,2, "
        "dan maksimal 100 kandidat CBF."
    )

@st.cache_data(show_spinner=False)
def _buf(data):
    """Membungkus bytes menjadi file-like object baru pada setiap pembacaan."""
    if isinstance(data, (bytes, bytearray)):
        return io.BytesIO(data)
    if hasattr(data, "getvalue"):          # UploadedFile Streamlit
        return io.BytesIO(data.getvalue())
    return data


@st.cache_data(show_spinner=False)
def load_data(ratings_bytes: bytes, books_bytes: bytes):
    ratings = pd.read_csv(
        _buf(ratings_bytes), sep=";", encoding="latin-1",
        dtype={"ISBN": str}, on_bad_lines="skip",
    )
    books = pd.read_csv(
        _buf(books_bytes), sep=";", encoding="latin-1",
        dtype={"ISBN": str}, on_bad_lines="skip",
    )

    # Bersihkan nama kolom dari tanda kutip dan spasi.
    ratings.columns = ratings.columns.astype(str).str.replace('"', "", regex=False).str.strip()
    books.columns = books.columns.astype(str).str.replace('"', "", regex=False).str.strip()

    # Menangani file rating yang terbaca sebagai satu kolom.
    if len(ratings.columns) == 1:
        ratings = pd.read_csv(
            _buf(ratings_bytes), encoding="latin-1",
            dtype=str, on_bad_lines="skip",
        )
        ratings.columns = ratings.columns.astype(str).str.replace('"', "", regex=False).str.strip()

    return ratings, books

    ratings.columns = ratings.columns.str.replace('"', '', regex=False).str.strip()
    books.columns = books.columns.str.replace('"', '', regex=False).str.strip()

    required_rating = {"User-ID", "ISBN", "Book-Rating"}
    required_book = {"ISBN", "Book-Title", "Book-Author"}
    if not required_rating.issubset(ratings.columns):
        raise ValueError("Kolom dataset rating harus memuat User-ID, ISBN, dan Book-Rating.")
    if not required_book.issubset(books.columns):
        raise ValueError("Kolom dataset buku harus memuat ISBN, Book-Title, dan Book-Author.")

    ratings["ISBN"] = ratings["ISBN"].astype(str).str.strip()
    books["ISBN"] = books["ISBN"].astype(str).str.strip()
    ratings["User-ID"] = pd.to_numeric(ratings["User-ID"], errors="coerce")
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    ratings["User-ID"] = ratings["User-ID"].astype(int)
    ratings = ratings[ratings["Book-Rating"] > 0].copy()

    # Menggunakan buku dengan minimal 5 rating, sesuai tahapan notebook.
    popular_isbns = ratings["ISBN"].value_counts()
    popular_isbns = popular_isbns[popular_isbns >= 5].index
    ratings = ratings[ratings["ISBN"].isin(popular_isbns)].copy()

    books = books.drop_duplicates(subset="ISBN").copy()
    for col in ["Book-Title", "Book-Author", "Subject"]:
        if col not in books.columns:
            books[col] = ""
        books[col] = books[col].fillna("").astype(str)

    # Subjek hasil pengayaan data digunakan jika tersedia.
    books["Content"] = (
        books["Book-Title"] + " " +
        books["Book-Author"] + " " +
        books["Subject"]
    ).str.strip()

    return ratings, books


@st.cache_resource(show_spinner="Menyiapkan model rekomendasi. Proses ini dapat memerlukan beberapa saat...")
def build_model(ratings, books):
    ratings = ratings.copy()
    books = books.copy()

    # Satu rating dari setiap pengguna dijadikan data testing.
    jumlah_rating = ratings.groupby("User-ID").size()
    eligible_users = jumlah_rating[jumlah_rating >= 2].index
    eligible_rows = ratings[ratings["User-ID"].isin(eligible_users)]

    test_data = (
        eligible_rows.groupby("User-ID", group_keys=False)
        .sample(n=1, random_state=2026)
    )
    train_data = ratings.drop(test_data.index).copy()

    # User-item matrix dalam format sparse agar penggunaan memori lebih efisien.
    user_codes, user_values = pd.factorize(train_data["User-ID"])
    item_codes, item_values = pd.factorize(train_data["ISBN"])
    user_ids = pd.Index(user_values)
    item_isbns = pd.Index(item_values)

    rating_matrix = sp.csr_matrix(
        (
            train_data["Book-Rating"].astype(float).to_numpy(),
            (user_codes, item_codes)
        ),
        shape=(len(user_ids), len(item_isbns))
    )

    # Cosine similarity antaritem untuk Collaborative Filtering.
    item_similarity = cosine_similarity(
        rating_matrix.T, dense_output=False
    ).tocsr()

    # Metadata buku yang ISBN-nya tersedia dalam data rating.
    metadata = books[books["ISBN"].isin(item_isbns)].copy()
    metadata = metadata.drop_duplicates("ISBN").reset_index(drop=True)

    # TF-IDF untuk Content-Based Filtering.
    tfidf = TfidfVectorizer(stop_words="english", max_features=100000)
    tfidf_matrix = tfidf.fit_transform(metadata["Content"])

    isbn_to_row = {isbn: i for i, isbn in enumerate(item_isbns)}
    isbn_to_meta = {isbn: i for i, isbn in enumerate(metadata["ISBN"])}
    user_to_row = {user_id: i for i, user_id in enumerate(user_ids)}

    return {
        "train_data": train_data,
        "test_data": test_data,
        "rating_matrix": rating_matrix,
        "item_similarity": item_similarity,
        "user_ids": user_ids,
        "item_isbns": item_isbns,
        "metadata": metadata,
        "tfidf_matrix": tfidf_matrix,
        "isbn_to_row": isbn_to_row,
        "isbn_to_meta": isbn_to_meta,
        "user_to_row": user_to_row,
    }


def recommend_books(user_id, model, top_n=5, alpha=0.4,
                    lambda_penalty=0.2, cbf_candidates=100):
    user_row = model["user_to_row"].get(user_id)
    if user_row is None:
        return pd.DataFrame()

    matrix = model["rating_matrix"]
    item_similarity = model["item_similarity"]
    item_isbns = model["item_isbns"]
    metadata = model["metadata"]
    tfidf_matrix = model["tfidf_matrix"]
    isbn_to_meta = model["isbn_to_meta"]

    user_vector = matrix.getrow(user_row)
    rated_item_indices = user_vector.indices
    rated_values = user_vector.data
    rated_isbns = set(item_isbns[rated_item_indices])

    if len(rated_item_indices) == 0:
        return pd.DataFrame()

    # 1. Collaborative Filtering: rata-rata tertimbang rating dari item serupa.
    cf_accumulator = {}
    for item_idx, rating in zip(rated_item_indices, rated_values):
        similarities = item_similarity.getrow(item_idx)
        for other_idx, similarity in zip(similarities.indices, similarities.data):
            other_isbn = item_isbns[other_idx]
            if other_isbn not in rated_isbns and similarity > 0:
                if other_isbn not in cf_accumulator:
                    cf_accumulator[other_isbn] = [0.0, 0.0]
                cf_accumulator[other_isbn][0] += similarity * rating
                cf_accumulator[other_isbn][1] += similarity

    cf_scores = {
        isbn: total / sim_sum
        for isbn, (total, sim_sum) in cf_accumulator.items()
        if sim_sum > 0
    }

    # 2. Content-Based Filtering: kemiripan konten dengan buku yang pernah dinilai.
    rated_meta_indices = [
        isbn_to_meta[isbn] for isbn in rated_isbns
        if isbn in isbn_to_meta
    ]
    rated_meta_isbns = [
        isbn for isbn in rated_isbns if isbn in isbn_to_meta
    ]
    if not rated_meta_indices:
        return pd.DataFrame()

    # Menyesuaikan urutan rating dengan indeks metadata.
    rating_lookup = dict(zip(item_isbns[rated_item_indices], rated_values))
    rated_meta_values = np.array([rating_lookup[x] for x in rated_meta_isbns])

    rated_content = tfidf_matrix[rated_meta_indices]
    similarities = (tfidf_matrix @ rated_content.T).toarray()

    similarity_sum = similarities.sum(axis=1)
    weighted_sum = similarities @ rated_meta_values
    confidence = similarity_sum / (similarity_sum + lambda_penalty)
    cbf_values = np.divide(
        weighted_sum, similarity_sum,
        out=np.zeros_like(weighted_sum, dtype=float),
        where=similarity_sum > 0
    ) * confidence

    cbf_scores = {}
    for isbn, meta_idx in isbn_to_meta.items():
        if isbn not in rated_isbns and similarity_sum[meta_idx] > 0:
            cbf_scores[isbn] = float(cbf_values[meta_idx])

    top_cbf = sorted(
        cbf_scores.items(), key=lambda x: x[1], reverse=True
    )[:cbf_candidates]

    # 3. Menggabungkan kandidat CF dan CBF.
    candidates = set(cf_scores).union(isbn for isbn, _ in top_cbf)
    if not candidates:
        return pd.DataFrame()

    result = pd.DataFrame({"ISBN": list(candidates)})
    result["CF-Score"] = result["ISBN"].map(cf_scores).fillna(0)
    result["CBF-Score"] = result["ISBN"].map(cbf_scores).fillna(0)

    # 4. Normalisasi Min-Max dan penghitungan skor Hybrid.
    for col in ["CF-Score", "CBF-Score"]:
        low, high = result[col].min(), result[col].max()
        result[col + "-Norm"] = (
            (result[col] - low) / (high - low) if high > low
            else 0.0
        )

    result["Hybrid-Score"] = (
        alpha * result["CF-Score-Norm"] +
        (1 - alpha) * result["CBF-Score-Norm"]
    )

    result = result.merge(
        metadata[["ISBN", "Book-Title", "Book-Author"]],
        on="ISBN", how="left"
    )
    result["Book-Title"] = result["Book-Title"].replace("", "Judul tidak tersedia")
    result["Book-Author"] = result["Book-Author"].replace("", "Penulis tidak tersedia")

    return result.sort_values(
        "Hybrid-Score", ascending=False
    ).head(top_n).reset_index(drop=True)


if not ratings_file or not books_file:
    st.info(
        "Silakan unggah kedua file dataset pada sidebar untuk mulai menggunakan sistem."
    )
    st.markdown("""
    **File yang diperlukan**
    - `BX-Book-Ratings.csv`
    - `BX-Books.csv`

    Pastikan file menggunakan dataset Book-Crossing dan format pemisah kolom titik koma (`;`).
    """)
    st.stop()

try:
    ratings, books = load_data(ratings_file.getvalue(), books_file.getvalue())
    if ratings.empty:
        st.error("Tidak ditemukan data rating setelah proses pembersihan dan penyaringan.")
        st.stop()

    with st.spinner("Memproses data dan membangun model..."):
        model = build_model(ratings, books)

except Exception as error:
    st.error(f"Terjadi kesalahan saat memproses dataset: {error}")
    st.stop()

# Ringkasan dataset.
st.subheader("Ringkasan Dataset")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Rating", f"{len(ratings):,}")
col2.metric("Jumlah Pengguna", f"{ratings['User-ID'].nunique():,}")
col3.metric("Jumlah Buku", f"{ratings['ISBN'].nunique():,}")
col4.metric("Buku pada Metadata", f"{len(model['metadata']):,}")

st.divider()
st.subheader("Temukan Rekomendasi Buku")
user_options = sorted(model["user_ids"].tolist())
selected_user = st.selectbox(
    "Pilih User ID",
    options=user_options,
    index=0,
    help="Pilih pengguna yang memiliki riwayat rating pada data training."
)

if st.button("Tampilkan Rekomendasi", type="primary", use_container_width=True):
    recommendations = recommend_books(
        user_id=selected_user,
        model=model,
        top_n=top_n,
        alpha=0.4,
        lambda_penalty=0.2,
        cbf_candidates=100
    )

    if recommendations.empty:
        st.warning(
            "Belum dapat menghasilkan rekomendasi untuk pengguna ini. "
            "Pengguna mungkin belum memiliki riwayat rating dengan metadata buku yang tersedia."
        )
    else:
        st.success(f"Berikut {len(recommendations)} rekomendasi buku untuk User ID {selected_user}.")
        for idx, row in recommendations.iterrows():
            with st.container(border=True):
                st.markdown(f"**{idx + 1}. {row['Book-Title']}**")
                st.write(f"Penulis: {row['Book-Author']}")
                st.caption(f"ISBN: {row['ISBN']}")
                st.progress(
                    float(np.clip(row["Hybrid-Score"], 0, 1)),
                    text=f"Hybrid Score: {row['Hybrid-Score']:.4f}"
                )
                with st.expander("Lihat rincian skor"):
                    a, b = st.columns(2)
                    a.metric("Collaborative Filtering", f"{row['CF-Score']:.4f}")
                    b.metric("Content-Based Filtering", f"{row['CBF-Score']:.4f}")

st.divider()
with st.expander("Tentang Model dan Parameter"):
    st.markdown("""
    **Konfigurasi model**
    - Collaborative Filtering: bobot 40%.
    - Content-Based Filtering: bobot 60%.
    - Lambda penalty: 0,2.
    - Kandidat dari Content-Based Filtering: maksimal 100 buku.
    - Rating 0 tidak digunakan dalam pemodelan.
    - Buku yang digunakan memiliki minimal 5 rating.

    Skor akhir dihitung dari kombinasi skor CF dan CBF yang telah dinormalisasi menggunakan Min-Max.
    """)
