import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import time
import pandas as pd
from datetime import datetime
from PIL import Image

# Library Checks
try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

st.set_page_config(page_title="Ultimate Kawakib Hilal Analyzer", page_icon="🌙", layout="wide")
st.title("🌙 Analisis Citra & Verifikasi Geometri Hilal Cerdas")
st.markdown("**Ultimate PoC V3.2 (Red Channel + Adaptive Threshold) - Kawakib Institute Semarang (ALICIA 2026)**")


# ==========================================
# FUNGSI UTAMA & UTILITY
# ==========================================
def calculate_mean_intensity(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    return np.mean(gray)


def calculate_snr(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    return round(mean_val / std_val, 2) if std_val > 0 else 0


def extract_channel(image_array, channel_mode):
    """
    Ekstraksi channel warna dari gambar RGB.
    Untuk hilal di langit senja, Red Channel umumnya memberikan kontras terbaik
    karena spektrum cahaya hilal cenderung ke merah/oranye.
    """
    if len(image_array.shape) == 2:
        return image_array.copy()  # Sudah grayscale
    if channel_mode == "Red Channel (Rekomendasi)":
        return image_array[:, :, 0].copy()
    elif channel_mode == "Green Channel":
        return image_array[:, :, 1].copy()
    elif channel_mode == "Blue Channel":
        return image_array[:, :, 2].copy()
    else:  # Grayscale (rata-rata)
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)


# ==========================================
# AUTO-SUGGEST LOGIC
# ==========================================
def suggest_params(current_intensity):
    if not os.path.exists("hilal_training_data.csv"):
        return None
    df = pd.read_csv("hilal_training_data.csv")
    if 'mean_intensity' not in df.columns:
        return None
    df['diff'] = abs(df['mean_intensity'] - current_intensity)
    best_match = df.loc[df['diff'].idxmin()]
    return best_match


def apply_params(params):
    st.session_state.f_size = int(params['filter_size'])
    st.session_state.p_low = float(params['p_low'])
    st.session_state.p_high = float(params['p_high'])
    st.session_state.blur = int(params['blur'])
    st.session_state.glue = int(params['glue'])
    st.session_state.min_a = int(params['min_a'])
    st.session_state.max_a = int(params['max_a'])
    st.session_state.max_c = float(params['max_c'])
    st.session_state.min_s = float(params['min_solidity'])
    st.rerun()


# ==========================================
# ENGINE PIPELINE
# ==========================================
def stack_video_pipeline(video_path, chunk_size, max_frames):
    cap = cv2.VideoCapture(video_path)
    frames_buffer, chunk_masters, frame_count = [], [], 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Ambil Red Channel dari video untuk konsistensi dengan pipeline gambar
        red_channel = frame[:, :, 2]  # BGR format: index 2 = Red
        frames_buffer.append(red_channel)
        frame_count += 1
        if len(frames_buffer) == chunk_size:
            ref_frame = np.float32(frames_buffer[0])
            aligned_frames = [ref_frame]
            for i in range(1, len(frames_buffer)):
                curr_frame = np.float32(frames_buffer[i])
                shift, _ = cv2.phaseCorrelate(curr_frame, ref_frame)
                dx, dy = shift
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned = cv2.warpAffine(curr_frame, M, (curr_frame.shape[1], curr_frame.shape[0]))
                aligned_frames.append(aligned)
            chunk_masters.append(np.median(np.array(aligned_frames), axis=0).astype(np.uint8))
            frames_buffer = []
    cap.release()
    if not chunk_masters:
        return None
    ref_frame = np.float32(chunk_masters[0])
    aligned_masters = [ref_frame]
    for i in range(1, len(chunk_masters)):
        curr_frame = np.float32(chunk_masters[i])
        shift, _ = cv2.phaseCorrelate(curr_frame, ref_frame)
        dx, dy = shift
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(curr_frame, M, (curr_frame.shape[1], curr_frame.shape[0]))
        aligned_masters.append(aligned)
    return np.median(np.array(aligned_masters), axis=0).astype(np.uint8)


def ultimate_pipeline_process(image_array, channel_mode, filter_size, p_low, p_high,
                               blur_ksize, glue_ksize, min_a, max_a, max_c, min_solidity,
                               threshold_mode, adapt_block, adapt_c):
    """
    Pipeline utama deteksi hilal.

    OPTIMASI V3.2:
    1. Ekstraksi Red Channel — lebih kontras untuk hilal di langit senja.
    2. Adaptive Thresholding — tahan terhadap gradasi cahaya latar belakang
       yang tidak merata (lebih baik dari OTSU untuk langit senja).
    """
    # ---- OPTIMASI 1: Ekstraksi Channel ----
    gray = extract_channel(image_array, channel_mode)

    # ---- Top-Hat Filter (menghilangkan gradasi langit) ----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # ---- Gaussian Blur (reduksi noise) ----
    smoothed = cv2.GaussianBlur(tophat, (blur_ksize, blur_ksize), 0) if blur_ksize > 0 else tophat

    # ---- Contrast Stretching (percentile) ----
    v_min, v_max = np.percentile(smoothed, (p_low, p_high))
    stretched = np.clip(
        ((smoothed - v_min) / (v_max - v_min + 1e-6)) * 255.0, 0, 255
    ).astype(np.uint8)

    # ---- OPTIMASI 2: Thresholding ----
    if threshold_mode == "Adaptive (Rekomendasi)":
        # Adaptive Thresholding: threshold dihitung secara lokal per blok area.
        # Lebih tahan gradasi langit senja dibanding OTSU global.
        # adapt_block harus ganjil dan >= 3
        block = adapt_block if adapt_block % 2 == 1 else adapt_block + 1
        block = max(block, 3)
        binary_mask = cv2.adaptiveThreshold(
            stretched, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block,
            C=adapt_c
        )
    else:
        # OTSU: threshold global (metode lama, sebagai fallback/pembanding)
        _, binary_mask = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ---- Dilation (lem piksel hilal yang terputus) ----
    binary_mask = cv2.dilate(
        binary_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glue_ksize, glue_ksize)),
        iterations=1
    )

    # ---- Deteksi Kontur Geometri ----
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_a < area < max_a:
            perimeter = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            if circ <= max_c and solidity <= min_solidity:
                valid_contours.append(cnt)

    # ---- Visualisasi ----
    visual = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(visual, contours, -1, (255, 0, 0), 1)   # Semua kontur: merah
    cv2.drawContours(visual, valid_contours, -1, (0, 255, 0), 3)  # Kontur valid hilal: hijau

    return (
        cv2.cvtColor(tophat, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB),
        visual,
        len(valid_contours) > 0
    )


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📁 Unggah Data & Config")
uploaded_file = st.sidebar.file_uploader(
    "Upload Media", type=['jpg', 'jpeg', 'png', 'avi', 'mp4', 'fits', 'fit']
)

# Mode hanya tersisa Kawakib Hybrid (YOLO dihapus karena belum ada model custom)
algo_mode = "🌙 Ultimate Kawakib V3 (Hybrid)"
st.sidebar.info("🌙 Mode aktif: **Ultimate Kawakib V3 (Hybrid)**")

# Inisialisasi session state default
if 'f_size' not in st.session_state:
    st.session_state.update({
        'f_size': 51, 'p_low': 90.0, 'p_high': 100.0,
        'blur': 11, 'glue': 5, 'min_a': 100, 'max_a': 50000,
        'max_c': 0.50, 'min_s': 0.6,
        'channel_mode': "Red Channel (Rekomendasi)",
        'threshold_mode': "Adaptive (Rekomendasi)",
        'adapt_block': 51, 'adapt_c': -5
    })

with st.sidebar.expander("🎨 Ekstraksi Channel Warna", expanded=True):
    st.caption("Red Channel terbaik untuk hilal di langit senja merah/oranye.")
    channel_mode = st.selectbox(
        "Channel Warna",
        ["Red Channel (Rekomendasi)", "Grayscale (Rata-rata RGB)", "Green Channel", "Blue Channel"],
        key="channel_mode"
    )

with st.sidebar.expander("🛠️ Parameter Tuning", expanded=True):
    f_size = st.slider("Ukuran Kernel", 5, 151, st.session_state.f_size, step=2, key="f_size")
    p_low = st.slider("Batas Bawah (%)", 0.0, 99.9, st.session_state.p_low, 0.1, key="p_low")
    p_high = st.slider("Batas Atas (%)", 0.0, 100.0, st.session_state.p_high, 0.1, key="p_high")
    blur = st.slider("Blur", 1, 31, st.session_state.blur, step=2, key="blur")
    glue = st.slider("Lem Piksel", 1, 21, st.session_state.glue, step=2, key="glue")
    min_a = st.slider("Min Area", 10, 5000, st.session_state.min_a, step=10, key="min_a")
    max_a = st.slider("Max Area", 1000, 150000, st.session_state.max_a, step=1000, key="max_a")
    max_c = st.slider("Maks. Circ", 0.01, 1.00, st.session_state.max_c, step=0.01, key="max_c")
    min_s = st.slider("Maks. Solidity", 0.1, 1.0, st.session_state.min_s, step=0.05, key="min_s")

with st.sidebar.expander("⚙️ Pengaturan Thresholding", expanded=True):
    st.caption("Adaptive lebih tahan gradasi cahaya langit dibanding OTSU.")
    threshold_mode = st.selectbox(
        "Metode Threshold",
        ["Adaptive (Rekomendasi)", "OTSU (Global, metode lama)"],
        key="threshold_mode"
    )
    if threshold_mode == "Adaptive (Rekomendasi)":
        adapt_block = st.slider("Adaptive Block Size", 11, 151, st.session_state.adapt_block, step=2, key="adapt_block")
        adapt_c = st.slider("Adaptive C (negatif = lebih ketat)", -20, 0, st.session_state.adapt_c, step=1, key="adapt_c")
    else:
        adapt_block = st.session_state.adapt_block
        adapt_c = st.session_state.adapt_c


# ==========================================
# PROSES UTAMA
# ==========================================
if uploaded_file is not None:
    # Load File
    if uploaded_file.name.lower().endswith(('.avi', '.mp4')):
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        t.write(uploaded_file.read())
        t.close()
        with st.spinner("⏳ Stacking & aligning frames video..."):
            image_array = stack_video_pipeline(t.name, 100, 300)
        if image_array is None:
            st.error("Gagal membaca frame video.")
    else:
        raw_image = np.array(Image.open(uploaded_file).convert("RGB"))
        image_array = raw_image

    if image_array is not None:
        # Auto-Suggest Logic
        curr_int = calculate_mean_intensity(image_array)
        suggestion = suggest_params(curr_int)
        if suggestion is not None:
            st.sidebar.warning(f"💡 Rekomendasi param ditemukan untuk kecerahan ~{int(curr_int)}.")
            if st.sidebar.button("🚀 Terapkan Parameter"):
                apply_params(suggestion)

        # ---- MAIN DISPLAY: Kawakib Hybrid ----
        start = time.time()
        top, stretch, det, valid = ultimate_pipeline_process(
            image_array, channel_mode,
            f_size, p_low, p_high, blur, glue,
            min_a, max_a, max_c, min_s,
            threshold_mode, adapt_block, adapt_c
        )
        elapsed = round(time.time() - start, 4)

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latensi", f"{elapsed}s")
        c2.metric("SNR Enhanced", calculate_snr(stretch))
        c3.metric("Kecerahan Raw", int(curr_int))
        if valid:
            c4.metric("Status Deteksi", "✅ HILAL TERDETEKSI")
        else:
            c4.metric("Status Deteksi", "❌ Tidak Terdeteksi")

        # Visualisasi Pipeline
        st.markdown("### 🔬 Visualisasi Pipeline")
        col_a, col_b, col_c = st.columns(3)
        col_a.image(top, caption=f"1. Top-Hat Filter ({channel_mode.split()[0]} Ch.)", use_container_width=True)
        col_b.image(stretch, caption="2. Contrast Stretching", use_container_width=True)
        col_c.image(det, caption="3. Geometric Detection (Hijau = Hilal)", use_container_width=True)

        if valid:
            st.success("🌙 Kandidat hilal terdeteksi berdasarkan filter geometri! Verifikasi visual diperlukan.")
        else:
            st.warning("Tidak ada kandidat hilal yang lolos filter geometri. Coba sesuaikan parameter di sidebar.")

        # Simpan Parameter
        st.markdown("---")
        if st.button("💾 Simpan Parameter ke Training Data"):
            new_row = pd.DataFrame([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mean_intensity": curr_int,
                "filter_size": f_size, "p_low": p_low, "p_high": p_high,
                "blur": blur, "glue": glue, "min_a": min_a, "max_a": max_a,
                "max_c": max_c, "min_solidity": min_s,
                "channel_mode": channel_mode, "threshold_mode": threshold_mode,
                "adapt_block": adapt_block, "adapt_c": adapt_c
            }])
            header = not os.path.exists("hilal_training_data.csv")
            new_row.to_csv("hilal_training_data.csv", mode='a', header=header, index=False)
            st.success("Parameter tersimpan ke hilal_training_data.csv!")

else:
    st.info("👈 Silakan upload gambar atau video hilal di sidebar untuk memulai analisis.")


# ==========================================
# PANEL ROADMAP YOLO (Informatif)
# ==========================================
st.markdown("---")
with st.expander("🤖 Roadmap: Deteksi Hilal dengan AI (YOLO Custom) — Belum Aktif"):
    st.markdown("""
    Mode **YOLO / Deep Learning** saat ini belum aktif karena membutuhkan model yang sudah
    di-*training* khusus dengan dataset gambar hilal. Model bawaan YOLOv8 tidak mengenal hilal.

    ### 📋 Langkah Membangun Model YOLO Custom Hilal

    | # | Langkah | Detail | Estimasi Waktu |
    |---|---|---|---|
    | 1 | **Kumpulkan Dataset** | Minimal 200–500 foto hilal dari berbagai kondisi (tipis, terang, berkabut, senja). Sumber: arsip BMKG, RHI, foto observasi. | 2–4 minggu |
    | 2 | **Anotasi / Labeling** | Tandai area hilal di setiap foto menggunakan [Roboflow](https://roboflow.com) (gratis) atau LabelImg. | 1–2 minggu |
    | 3 | **Training Model** | Fine-tune YOLOv8 dengan dataset. Bisa gratis di Google Colab dengan GPU. | 1–3 hari |
    | 4 | **Integrasi ke App** | Simpan hasil sebagai `hilal_yolo_custom.pt` dan load di aplikasi ini. | Instan |

    ### 💡 Tips
    - **Roboflow** otomatis men-*augment* dataset (rotasi, kecerahan, flip) sehingga 200 foto bisa menjadi ribuan variasi.
    - Sertakan juga foto **non-hilal** (awan, pesawat, bintang terang) sebagai *negative samples* agar model tidak salah deteksi.
    - Target awal: akurasi ≥ 80% di kondisi normal, ≥ 60% di kondisi remang/berkabut.

    > Setelah model custom siap, hubungi tim pengembang Kawakib Institute untuk integrasi ke versi berikutnya.
    """)