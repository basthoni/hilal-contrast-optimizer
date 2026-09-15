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
st.markdown("**Ultimate PoC V3.1 (Restored & Auto-Suggest) - Kawakib Institute Semarang (ALICIA 2026)**")


# ==========================================
# FUNGSI UTAMA & UTILITY
# ==========================================
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    if HAS_YOLO: return YOLO('yolov8n.pt')
    return None


def calculate_mean_intensity(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    return np.mean(gray)


def calculate_snr(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    return round(mean_val / std_val, 2) if std_val > 0 else 0


# ==========================================
# AUTO-SUGGEST LOGIC
# ==========================================
def suggest_params(current_intensity):
    if not os.path.exists("hilal_training_data.csv"): return None
    df = pd.read_csv("hilal_training_data.csv")
    if 'mean_intensity' not in df.columns: return None

    # Mencari yang paling mendekati kecerahan saat ini
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
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_buffer.append(gray)
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
    if not chunk_masters: return None
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


def ultimate_pipeline_process(image_array, filter_size, p_low, p_high, blur_ksize, glue_ksize, min_a, max_a, max_c,
                              min_solidity):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filter_size, filter_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    smoothed = cv2.GaussianBlur(tophat, (blur_ksize, blur_ksize), 0) if blur_ksize > 0 else tophat
    v_min, v_max = np.percentile(smoothed, (p_low, p_high))
    stretched = np.clip(((smoothed - v_min) / (v_max - v_min + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (glue_ksize, glue_ksize)),
                             iterations=1)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_a < area < max_a:
            perimeter = cv2.arcLength(cnt, True)
            circ = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            solidity = float(area) / cv2.contourArea(cv2.convexHull(cnt)) if cv2.contourArea(
                cv2.convexHull(cnt)) > 0 else 0
            if circ <= max_c and solidity <= min_solidity: valid_contours.append(cnt)

    visual = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(visual, contours, -1, (255, 0, 0), 1)
    cv2.drawContours(visual, valid_contours, -1, (0, 255, 0), 3)
    return cv2.cvtColor(tophat, cv2.COLOR_GRAY2RGB), cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB), visual, len(
        valid_contours) > 0


def yolo_real_pipeline(image_array):
    model = load_yolo_model()
    if model is None: return None, 0, []
    img_in = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB) if len(image_array.shape) == 2 else image_array.copy()
    results = model(img_in, verbose=False)
    classes = [f"{model.names[int(box.cls[0])].upper()} ({float(box.conf[0]) * 100:.1f}%)" for box in results[0].boxes]
    return results[0].plot(), len(results[0].boxes), classes


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("📁 Unggah Data & Config")
uploaded_file = st.sidebar.file_uploader("Upload Media", type=['jpg', 'jpeg', 'png', 'avi', 'mp4', 'fits', 'fit'])

algo_mode = st.sidebar.radio("Pilih Algoritma:", (
"🌙 Ultimate Kawakib V3 (Hybrid)", "📦 Konvensional (YOLO Asli)", "🍌 Uji Ekstrem (YOLO + Enhancement)"))

if 'f_size' not in st.session_state:
    st.session_state.update(
        {'f_size': 51, 'p_low': 90.0, 'p_high': 100.0, 'blur': 11, 'glue': 5, 'min_a': 100, 'max_a': 50000,
         'max_c': 0.50, 'min_s': 0.6})

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

# ==========================================
# PROSES UTAMA
# ==========================================
if uploaded_file is not None:
    # Load File
    if uploaded_file.name.lower().endswith(('.avi', '.mp4')):
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".avi");
        t.write(uploaded_file.read());
        t.close()
        image_array = stack_video_pipeline(t.name, 100, 300)
    else:
        image_array = np.array(Image.open(uploaded_file))

    if image_array is not None:
        # Auto-Suggest Logic
        curr_int = calculate_mean_intensity(image_array)
        suggestion = suggest_params(curr_int)
        if suggestion is not None:
            st.sidebar.warning(f"💡 Rekomendasi param ditemukan untuk kecerahan ~{int(curr_int)}.")
            if st.sidebar.button("🚀 Terapkan Parameter"):
                apply_params(suggestion)

        # UI 3-Column Display for Hybrid
        if algo_mode == "🌙 Ultimate Kawakib V3 (Hybrid)":
            start = time.time()
            top, stretch, det, valid = ultimate_pipeline_process(image_array, f_size, p_low, p_high, blur, glue, min_a,
                                                                 max_a, max_c, min_s)

            c1, c2, c3 = st.columns(3)
            c1.metric("Latensi", f"{round(time.time() - start, 4)}s")
            c2.metric("SNR Enhanced", calculate_snr(stretch))
            c3.metric("Kecerahan", int(curr_int))

            st.markdown("### Visualisasi Pipeline")
            col_a, col_b, col_c = st.columns(3)
            col_a.image(top, caption="1. Top-Hat Filter", use_container_width=True)
            col_b.image(stretch, caption="2. Contrast Stretching", use_container_width=True)
            col_c.image(det, caption="3. Geometric Detection", use_container_width=True)

            if st.button("💾 Simpan Parameter"):
                pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "mean_intensity": curr_int,
                    "filter_size": f_size, "p_low": p_low, "p_high": p_high,
                    "blur": blur, "glue": glue, "min_a": min_a, "max_a": max_a,
                    "max_c": max_c, "min_solidity": min_s
                }]).to_csv("hilal_training_data.csv", mode='a', header=not os.path.exists("hilal_training_data.csv"),
                           index=False)
                st.success("Tersimpan!")

        elif algo_mode == "📦 Konvensional (YOLO Asli)":
            start = time.time()
            vis, count, cls = yolo_real_pipeline(image_array)
            st.metric("Latensi YOLO", f"{round(time.time() - start, 4)}s")
            st.metric("SNR Raw", calculate_snr(image_array))
            st.warning(f"YOLO melihat: {', '.join(cls) if cls else 'Buta'}")
            st.image(vis, use_container_width=True)

        elif algo_mode == "🍌 Uji Ekstrem (YOLO + Enhancement)":
            _, stretch, _, _ = ultimate_pipeline_process(image_array, f_size, p_low, p_high, blur, glue, min_a, max_a,
                                                         max_c, min_s)
            start = time.time()
            vis, count, cls = yolo_real_pipeline(stretch)

            st.metric("Latensi (V2+YOLO)", f"{round(time.time() - start, 4)}s")
            st.metric("SNR Enhanced", calculate_snr(stretch))
            col1, col2 = st.columns(2)
            col1.image(stretch, caption="Input Enhancement", use_container_width=True)
            col2.image(vis, caption=f"Deteksi YOLO: {', '.join(cls) if cls else 'Buta'}", use_container_width=True)