"""
Sistem Rekomendasi Buku (Hybrid: Collaborative Filtering + Content-Based Filtering)
Dataset: Book-Crossing (BX-Book-Ratings.csv dan BX-Books.csv)

Alur sistem:
1. CBF   : profil TF-IDF pengguna dibandingkan dengan seluruh buku -> 100 kandidat teratas.
2. CF    : user-based CF (mean-centered cosine) memprediksi rating tiap kandidat.
3. Hybrid: skor = 0,4 x CF + 0,6 x CBF.
4. MMR   : re-ranking dengan lambda 0,2 untuk mengurangi redundansi (diversifikasi) Top-N.
"""

import hashlib
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ============================================================
# KONFIGURASI HALAMAN DAN PARAMETER
# ============================================================
st.set_page_config(
    page_title="Sistem Rekomendasi Buku",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

W_CF = 0.4                # bobot Collaborative Filtering
W_CBF = 0.6               # bobot Content-Based Filtering
LAMBDA = 0.2              # parameter MMR (bobot penalti redundansi)
N_CANDIDATES = 100        # jumlah kandidat dari CBF
K_NEIGHBORS = 50          # jumlah tetangga terdekat pada CF
MIN_OVERLAP = 2           # minimal buku yang sama-sama dinilai agar kemiripan dihitung
SIGNIFICANCE_CAP = 20     # kemiripan dilemahkan jika irisan buku < nilai ini
BAYES_PRIOR = 5           # kekuatan prior untuk rata-rata rating buku (fallback CF)
MIN_USER_RATINGS = 5      # pengguna dengan rating eksplisit < nilai ini dibuang
MIN_BOOK_RATINGS = 5      # buku dengan rating eksplisit < nilai ini dibuang

COLUMN_ALIASES = {
    "user-id": "User-ID", "userid": "User-ID", "user_id": "User-ID",
    "isbn": "ISBN",
    "book-rating": "Book-Rating", "bookrating": "Book-Rating", "rating": "Book-Rating",
    "book-title": "Book-Title", "title": "Book-Title",
    "book-author": "Book-Author", "author": "Book-Author",
    "year-of-publication": "Year-Of-Publication",
    "publisher": "Publisher",
    "image-url-m": "Image-URL-M",
    "subject": "Subject", "subjects": "Subject", "book-subject": "Subject",
    "category": "Subject", "categories": "Subject", "genre": "Subject",
}


# ============================================================
# FUNGSI BANTU PEMBACAAN DATA
# ============================================================
def _buf(data):
    """Membungkus bytes menjadi file-like object baru pada setiap pembacaan."""
    if isinstance(data, (bytes, bytearray)):
        return io.BytesIO(data)
    if hasattr(data, "getvalue"):
        return io.BytesIO(data.getvalue())
    return data


def _try_read(data, sep):
    """Mencoba membaca CSV dengan engine C, lalu engine Python bila gagal."""
    opts = dict(sep=sep, encoding="latin-1", dtype=str, on_bad_lines="skip")
    try:
        return pd.read_csv(_buf(data), low_memory=False, **opts)
    except Exception:
        try:
            return pd.read_csv(_buf(data), engine="python", **opts)
        except Exception:
            return None


def _read_table(data):
    """Membaca tabel dengan mencoba beberapa pemisah kolom (; , tab)."""
    for sep in (";", ",", "\t"):
        frame = _try_read(data, sep)
        if frame is not None and frame.shape[1] > 1:
            return frame
    raise ValueError(
        "Format file tidak sesuai. Pastikan file berformat CSV dengan pemisah kolom ';'."
    )


def _standardize_columns(frame):
    """Membersihkan nama kolom dan menyeragamkannya dengan nama standar Book-Crossing."""
    cols = frame.columns.astype(str).str.replace('"', "", regex=False).str.strip()
    frame.columns = [COLUMN_ALIASES.get(c.lower(), c) for c in cols]
    return frame.loc[:, ~frame.columns.duplicated()]


def _clean_series(series):
    return series.fillna("").astype(str).str.replace('"', "", regex=False).str.strip()


def _normalize_text(series):
    """Huruf kecil, hapus tanda baca, rapikan spasi."""
    return (
        series.str.lower()
        .str.replace(r"[\W_]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ============================================================
# PERSIAPAN DATA DAN PEMBENTUKAN MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def prepare_model(ratings_bytes: bytes, books_bytes: bytes) -> dict:
    """Membaca dataset, melakukan preprocessing, dan membangun matriks CF serta TF-IDF CBF."""
    ratings = _standardize_columns(_read_table(ratings_bytes))
    books = _standardize_columns(_read_table(books_bytes))

    # Antisipasi jika kedua file tertukar saat diunggah.
    if "Book-Title" in ratings.columns and "Book-Rating" in books.columns:
        ratings, books = books, ratings

    required_rating = ["User-ID", "ISBN", "Book-Rating"]
    required_book = ["ISBN", "Book-Title", "Book-Author"]
    missing_r = [c for c in required_rating if c not in ratings.columns]
    missing_b = [c for c in required_book if c not in books.columns]
    if missing_r:
        raise ValueError(f"Kolom pada file rating tidak ditemukan: {', '.join(missing_r)}.")
    if missing_b:
        raise ValueError(f"Kolom pada file buku tidak ditemukan: {', '.join(missing_b)}.")

    # ---------- Preprocessing rating ----------
    ratings = ratings[required_rating].copy()
    for col in required_rating:
        ratings[col] = _clean_series(ratings[col])
    ratings = ratings[(ratings["User-ID"] != "") & (ratings["ISBN"] != "")]
    ratings["ISBN"] = ratings["ISBN"].str.upper()
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings = ratings.dropna(subset=["Book-Rating"])
    n_raw = len(ratings)
    ratings = ratings[ratings["Book-Rating"] > 0]            # hanya rating eksplisit (1-10)
    ratings = ratings.drop_duplicates(["User-ID", "ISBN"])

    # ---------- Preprocessing buku ----------
    optional = [c for c in ["Subject", "Year-Of-Publication", "Publisher", "Image-URL-M"]
                if c in books.columns]
    books = books[required_book + optional].copy()
    for col in required_book + optional:
        books[col] = _clean_series(books[col])
    books["ISBN"] = books["ISBN"].str.upper()
    books = books[(books["ISBN"] != "") & (books["Book-Title"] != "")]
    books = books.drop_duplicates("ISBN")

    ratings = ratings[ratings["ISBN"].isin(set(books["ISBN"]))]

    # ---------- Penyaringan pengguna dan buku yang terlalu jarang ----------
    for _ in range(3):
        before = len(ratings)
        ratings = ratings[
            ratings.groupby("User-ID")["ISBN"].transform("size") >= MIN_USER_RATINGS
        ]
        ratings = ratings[
            ratings.groupby("ISBN")["User-ID"].transform("size") >= MIN_BOOK_RATINGS
        ]
        if len(ratings) == before:
            break
    if ratings.empty:
        raise ValueError(
            "Tidak ada data yang tersisa setelah penyaringan. "
            "Periksa isi file atau turunkan MIN_USER_RATINGS / MIN_BOOK_RATINGS."
        )

    # ---------- Matriks pengguna x buku (sparse) ----------
    u_codes, u_unique = pd.factorize(ratings["User-ID"])
    b_codes, b_unique = pd.factorize(ratings["ISBN"])
    n_users, n_items = len(u_unique), len(b_unique)

    R = sp.csr_matrix(
        (ratings["Book-Rating"].to_numpy(dtype=np.float32), (u_codes, b_codes)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    counts = np.diff(R.indptr)
    user_means = np.asarray(R.sum(axis=1)).ravel() / np.maximum(counts, 1)

    Rc = R.copy()                                              # rating terpusat per pengguna
    Rc.data = Rc.data - np.repeat(user_means, counts).astype(np.float32)
    Rn = normalize(Rc, norm="l2", axis=1).tocsr()              # baris ternormalisasi (cosine)
    B = R.copy()                                               # matriks biner (sudah menilai)
    B.data[:] = 1.0

    item_counts = np.asarray(B.sum(axis=0)).ravel()
    item_sums = np.asarray(R.sum(axis=0)).ravel()
    global_mean = float(R.data.mean())
    item_bayes = (item_sums + BAYES_PRIOR * global_mean) / (item_counts + BAYES_PRIOR)

    # ---------- TF-IDF konten buku ----------
    meta = books.set_index("ISBN").reindex(b_unique).fillna("")
    title = _normalize_text(meta["Book-Title"])
    author = _normalize_text(meta["Book-Author"])
    author_token = author.str.replace(" ", "", regex=False)
    subject = _normalize_text(meta["Subject"]) if "Subject" in meta.columns else ""
    corpus = (title + " " + author + " " + author_token + " " + subject).str.strip()

    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=60000, sublinear_tf=True
    )
    tfidf = vectorizer.fit_transform(corpus).astype(np.float32).tocsr()

    top_idx = np.argsort(-counts)[:20]
    return {
        "R": R, "Rc": Rc, "Rn": Rn, "B": B,
        "tfidf": tfidf,
        "user_means": user_means,
        "item_bayes": item_bayes,
        "item_counts": item_counts,
        "user_index": {uid: i for i, uid in enumerate(u_unique)},
        "meta": meta.reset_index().rename(columns={"index": "ISBN"}),
        "top_users": [str(u_unique[i]) for i in top_idx],
        "has_subject": "Subject" in meta.columns,
        "stats": {
            "n_raw": int(n_raw),
            "n_ratings": int(R.nnz),
            "n_users": int(n_users),
            "n_items": int(n_items),
            "density": float(R.nnz / (n_users * n_items)),
            "global_mean": global_mean,
        },
    }


# ============================================================
# COLLABORATIVE FILTERING (USER-BASED)
# ============================================================
def user_based_cf(model: dict, u: int):
    """
    Memprediksi rating seluruh buku untuk pengguna u.
    Kemiripan: cosine pada rating yang dipusatkan, dilemahkan oleh jumlah irisan buku.
    Mengembalikan (prediksi, jumlah_tetangga_pendukung) atau (None, None).
    """
    Rn, Rc, B = model["Rn"], model["Rc"], model["B"]

    sims = (Rn @ Rn[u].T).toarray().ravel()
    overlap = (B @ B[u].T).toarray().ravel()
    sims = sims * np.minimum(overlap, SIGNIFICANCE_CAP) / SIGNIFICANCE_CAP
    sims[overlap < MIN_OVERLAP] = 0.0
    sims[u] = 0.0

    pos = np.flatnonzero(sims > 0)
    if pos.size == 0:
        return None, None
    if pos.size > K_NEIGHBORS:
        pos = pos[np.argpartition(sims[pos], -K_NEIGHBORS)[-K_NEIGHBORS:]]
    w = sims[pos].astype(np.float32)

    numerator = Rc[pos].T @ w
    denominator = B[pos].T @ np.abs(w)
    support = np.asarray(B[pos].sum(axis=0)).ravel()

    pred = np.full(Rc.shape[1], np.nan, dtype=np.float32)
    ok = denominator > 0
    pred[ok] = model["user_means"][u] + numerator[ok] / denominator[ok]
    return pred, support


# ============================================================
# MESIN REKOMENDASI HYBRID
# ============================================================
def recommend(model: dict, user_id: str, top_n: int) -> pd.DataFrame:
    """Menghasilkan Top-N rekomendasi hybrid untuk satu pengguna."""
    u = model["user_index"][user_id]
    R, tfidf = model["R"], model["tfidf"]

    # ---------- 1. CBF: profil pengguna dan 100 kandidat ----------
    row = R[u]
    rated_items = row.indices
    profile = row @ tfidf
    if profile.nnz == 0:
        return pd.DataFrame()

    sim_cbf = cosine_similarity(profile, tfidf).ravel()
    sim_cbf[rated_items] = -1.0                               # buang buku yang sudah dinilai

    n_valid = int((sim_cbf > 0).sum())
    n_cand = min(N_CANDIDATES, n_valid)
    if n_cand == 0:
        return pd.DataFrame()
    cand = np.argpartition(sim_cbf, -n_cand)[-n_cand:]
    cand = cand[np.argsort(-sim_cbf[cand])]

    # ---------- 2. CF: prediksi rating kandidat ----------
    pred_full, support_full = user_based_cf(model, u)
    if pred_full is None:
        cf_pred = np.full(n_cand, np.nan, dtype=np.float32)
        cf_support = np.zeros(n_cand)
    else:
        cf_pred = pred_full[cand]
        cf_support = support_full[cand]

    from_neighbors = ~np.isnan(cf_pred)
    cf_value = np.where(from_neighbors, cf_pred, model["item_bayes"][cand])  # cadangan: rata-rata buku
    cf_value = np.clip(cf_value, 1.0, 10.0)
    cf_score = (cf_value - 1.0) / 9.0

    # ---------- 3. Normalisasi CBF dan penggabungan hybrid ----------
    cbf_raw = sim_cbf[cand]
    cbf_score = (cbf_raw - cbf_raw.min()) / max(float(np.ptp(cbf_raw)), 1e-9)
    hybrid = W_CF * cf_score + W_CBF * cbf_score

    # ---------- 4. MMR: pemilihan Top-N yang beragam ----------
    item_sim = cosine_similarity(tfidf[cand])
    selected, remaining = [], list(range(n_cand))
    while remaining and len(selected) < top_n:
        if selected:
            redundancy = item_sim[np.ix_(remaining, selected)].max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        mmr = (1.0 - LAMBDA) * hybrid[remaining] - LAMBDA * redundancy
        best = remaining[int(np.argmax(mmr))]
        selected.append(best)
        remaining.remove(best)

    sel = np.array(selected)
    items = cand[sel]
    meta = model["meta"].iloc[items]

    result = pd.DataFrame({
        "Peringkat": np.arange(1, len(items) + 1),
        "Judul": meta["Book-Title"].to_numpy(),
        "Penulis": meta["Book-Author"].to_numpy(),
        "Skor Hybrid": hybrid[sel],
        "Skor CF": cf_score[sel],
        "Skor CBF": cbf_score[sel],
        "Prediksi Rating": cf_value[sel],
        "Dasar CF": np.where(from_neighbors[sel], "Tetangga pengguna", "Rata-rata buku"),
        "Tetangga Pendukung": cf_support[sel].astype(int),
        "ISBN": meta["ISBN"].to_numpy(),
    })
    for col in ["Subject", "Year-Of-Publication", "Publisher", "Image-URL-M"]:
        if col in meta.columns:
            result[col] = meta[col].to_numpy()
    return result


def user_history(model: dict, user_id: str) -> pd.DataFrame:
    """Riwayat rating pengguna, diurutkan dari rating tertinggi."""
    u = model["user_index"][user_id]
    row = model["R"][u]
    meta = model["meta"].iloc[row.indices]
    history = pd.DataFrame({
        "Judul": meta["Book-Title"].to_numpy(),
        "Penulis": meta["Book-Author"].to_numpy(),
        "Rating": row.data.astype(int),
        "ISBN": meta["ISBN"].to_numpy(),
    })
    return history.sort_values(["Rating", "Judul"], ascending=[False, True]).reset_index(drop=True)


# ============================================================
# ANTARMUKA PENGGUNA
# ============================================================
st.title("📚 Sistem Rekomendasi Buku")
st.caption(
    "Hybrid Recommendation System dengan Collaborative Filtering "
    "dan Content-Based Filtering."
)
st.markdown(
    """
    Sistem menggabungkan dua pendekatan:
    - **Collaborative Filtering (CF):** menggunakan pola rating pengguna terhadap buku.
    - **Content-Based Filtering (CBF):** menggunakan kemiripan judul, penulis, dan subjek buku melalui TF-IDF.
    """
)

with st.sidebar:
    st.header("Pengaturan Sistem")
    ratings_file = st.file_uploader(
        "Upload BX-Book-Ratings.csv", type=["csv"], key="ratings_file"
    )
    books_file = st.file_uploader(
        "Upload BX-Books.csv", type=["csv"], key="books_file"
    )
    top_n = st.selectbox("Jumlah rekomendasi", [5, 10, 15, 20], index=0)
    st.divider()
    st.caption("Bobot CF: 40% | Bobot CBF: 60% | Lambda: 0,2 | Kandidat CBF: 100")


def resolve_source(upload, filename):
    """Memakai file unggahan; jika tidak ada, memakai file di repository (bila tersedia)."""
    if upload is not None:
        return upload.getvalue()
    path = BASE_DIR / filename
    return path.read_bytes() if path.exists() else None


ratings_src = resolve_source(ratings_file, "BX-Book-Ratings.csv")
books_src = resolve_source(books_file, "BX-Books.csv")

if ratings_src is None or books_src is None:
    st.info(
        "Silakan unggah **BX-Book-Ratings.csv** dan **BX-Books.csv** pada sidebar "
        "(atau letakkan kedua file tersebut di repository, sejajar dengan app.py)."
    )
    st.stop()

try:
    with st.spinner("Membaca dataset dan membangun model (hanya sekali per dataset)..."):
        model = prepare_model(ratings_src, books_src)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses dataset: {e}")
    st.exception(e)
    st.stop()

stats = model["stats"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pengguna", f"{stats['n_users']:,}")
c2.metric("Buku", f"{stats['n_items']:,}")
c3.metric("Rating eksplisit", f"{stats['n_ratings']:,}")
c4.metric("Kepadatan matriks", f"{stats['density'] * 100:.3f}%")

st.subheader("Pilih Pengguna")
user_input = st.text_input(
    "User-ID",
    value=model["top_users"][0],
    help="Masukkan User-ID yang terdapat pada data setelah penyaringan.",
).strip()

with st.expander("Contoh User-ID dengan rating terbanyak"):
    st.write(", ".join(model["top_users"]))

if user_input not in model["user_index"]:
    st.warning(
        f"User-ID **{user_input}** tidak ditemukan. Pengguna harus memiliki minimal "
        f"{MIN_USER_RATINGS} rating eksplisit (nilai 1-10) pada buku yang memiliki minimal "
        f"{MIN_BOOK_RATINGS} rating."
    )
    st.stop()

try:
    with st.spinner("Menghitung rekomendasi..."):
        recs = recommend(model, user_input, top_n)
        history = user_history(model, user_input)
except Exception as e:
    st.error(f"Terjadi kesalahan saat menghitung rekomendasi: {e}")
    st.exception(e)
    st.stop()

tab_rec, tab_hist, tab_info = st.tabs(["Rekomendasi", "Riwayat Rating", "Metode dan Data"])

with tab_rec:
    st.subheader(f"Top-{top_n} Rekomendasi untuk User-ID {user_input}")
    if recs.empty:
        st.warning("Rekomendasi tidak dapat dibentuk karena profil konten pengguna kosong.")
    else:
        column_config = {
            "Skor Hybrid": st.column_config.ProgressColumn(
                "Skor Hybrid", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Skor CF": st.column_config.NumberColumn("Skor CF", format="%.3f"),
            "Skor CBF": st.column_config.NumberColumn("Skor CBF", format="%.3f"),
            "Prediksi Rating": st.column_config.NumberColumn("Prediksi Rating", format="%.2f"),
        }
        if "Image-URL-M" in recs.columns:
            column_config["Image-URL-M"] = st.column_config.ImageColumn("Sampul")

        st.dataframe(recs, hide_index=True, width="stretch", column_config=column_config)
        st.download_button(
            "Unduh hasil rekomendasi (CSV)",
            data=recs.to_csv(index=False).encode("utf-8"),
            file_name=f"rekomendasi_user_{user_input}.csv",
            mime="text/csv",
        )
        st.caption(
            "Skor Hybrid = 0,4 x Skor CF + 0,6 x Skor CBF. Urutan akhir ditentukan oleh "
            "re-ranking MMR (lambda 0,2), sehingga dapat berbeda tipis dari urutan skor murni. "
            "Dasar CF 'Rata-rata buku' berarti belum ada tetangga yang menilai buku tersebut."
        )

with tab_hist:
    st.subheader(f"Riwayat Rating User-ID {user_input}")
    st.write(
        f"Pengguna ini menilai **{len(history)}** buku dengan rata-rata rating "
        f"**{history['Rating'].mean():.2f}**."
    )
    st.dataframe(history, hide_index=True, width="stretch")

with tab_info:
    st.subheader("Ringkasan Data")
    st.write(
        f"- Rating eksplisit awal (nilai > 0 dan > 0 sebelum penyaringan): "
        f"**{stats['n_raw']:,}** baris termasuk rating implisit (0).\n"
        f"- Setelah penyaringan (pengguna >= {MIN_USER_RATINGS} rating, "
        f"buku >= {MIN_BOOK_RATINGS} rating): **{stats['n_ratings']:,}** rating, "
        f"**{stats['n_users']:,}** pengguna, **{stats['n_items']:,}** buku.\n"
        f"- Rata-rata rating global: **{stats['global_mean']:.2f}**.\n"
        f"- Kolom subjek pada TF-IDF: "
        f"**{'digunakan' if model['has_subject'] else 'tidak tersedia pada file buku'}**."
    )
    st.subheader("Metode")
    st.markdown(
        f"""
        1. **CBF:** profil pengguna dibentuk dari rata-rata TF-IDF buku yang dinilai
           (dibobot rating), lalu dibandingkan dengan seluruh buku menggunakan cosine
           similarity. Diambil **{N_CANDIDATES}** kandidat teratas yang belum pernah dinilai.
        2. **CF (user-based):** kemiripan antarpengguna dihitung dengan cosine pada rating
           yang dipusatkan, dilemahkan jika irisan buku sedikit. Prediksi rating dihitung
           dari **{K_NEIGHBORS}** tetangga terdekat.
        3. **Hybrid:** skor akhir = **{W_CF:.1f} x CF + {W_CBF:.1f} x CBF**
           (CF dinormalisasi dari skala 1-10, CBF dinormalisasi min-max pada kandidat).
        4. **Re-ranking MMR:** Top-N dipilih dengan penalti redundansi lambda = **{LAMBDA}**
           agar rekomendasi tidak terlalu seragam.
        """
    )
