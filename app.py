import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Klasifikasi Emosi Wajah - CNN", layout="wide")

BASE = Path(__file__).resolve().parent
HASIL = BASE / "hasil"
MODEL_PATH = HASIL / "model_emosi_cnn.keras"
IMG_SIZE = 48


# ------------------------------------------------------------------ helper
def read_json(name):
    p = HASIL / name
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_cm():
    p = HASIL / "confusion_matrix.npy"
    return np.load(p) if p.exists() else None


def lengkapi_info(info, met):
    """Melengkapi info dataset yang kuncinya hilang (class_names, counts, total)."""
    info = dict(info or {})
    kelas = info.get("class_names")
    if not kelas and met:
        kelas = list(met.get("per_class", {}).keys())
    if not kelas and info.get("split"):
        kelas = [r["Emosi"] for r in info["split"]]
    if not kelas and info.get("counts"):
        kelas = list(info["counts"].keys())
    info["class_names"] = kelas or []
    if not info.get("counts") and info.get("split"):
        info["counts"] = {r["Emosi"]: int(r["Latih"] + r["Validasi"] + r["Uji"]) for r in info["split"]}
    if not info.get("total") and info.get("counts"):
        info["total"] = int(sum(info["counts"].values()))
    info.setdefault("img_size", IMG_SIZE)
    return info


def show_file(name, width=None):
    p = HASIL / name
    if p.exists():
        img = Image.open(p)
        if width:
            st.image(img, width=width)
        else:
            st.image(img)
    else:
        st.info("Gambar " + name + " belum ada di folder hasil. Jalankan notebook sampai selesai.")


def heatmap(ax, data, labels, fmt):
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label asli")
    top = data.max() if data.max() > 0 else 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            warna = "white" if data[i, j] > top * 0.6 else "black"
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                    color=warna, fontsize=8)
    return im


@st.cache_resource
def load_model():
    from tensorflow import keras
    return keras.models.load_model(str(MODEL_PATH), compile=False)


def gradcam(x, last_conv):
    import tensorflow as tf
    from tensorflow import keras
    model = load_model()
    gm = keras.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = gm(x, training=False)
        idx = int(tf.argmax(preds[0]))
        score = preds[:, idx]
    grads = tape.gradient(score, conv_out)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * w[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
    return cam


# ------------------------------------------------------------------ muat hasil
hist = read_json("history.json")
tinfo = read_json("train_info.json")
met = read_json("metrics.json")
cm = read_cm()
info = lengkapi_info(read_json("info_dataset.json"), met)
class_names = info["class_names"]

# ------------------------------------------------------------------ sidebar debug
with st.sidebar:
    st.header("Info debug")
    st.write("Folder app:", str(BASE))
    st.write("Folder hasil ada:", HASIL.exists())
    if HASIL.exists():
        st.write(sorted(p.name for p in HASIL.iterdir()))
    st.write("Model ada:", MODEL_PATH.exists())

# ------------------------------------------------------------------ header
st.title("Klasifikasi Citra Emosi Wajah Menggunakan CNN")
st.write(
    "Dashboard hasil praktikum Convolutional Neural Network (CNN) untuk mengenali emosi wajah. "
    "Model dilatih pada dataset Facial Emotion Recognition dari Kaggle, lalu dievaluasi dan dapat "
    "dipakai untuk memprediksi gambar baru secara langsung."
)

if not class_names:
    st.error("Daftar kelas tidak ditemukan di folder: " + str(HASIL))
    st.write("Pastikan app.py berada di folder proyek yang sama dengan folder 'hasil', "
             "dan notebook sudah dijalankan sampai selesai.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Ringkasan Dataset", "Hasil Pelatihan", "Evaluasi Model", "Visualisasi", "Prediksi Gambar Baru"]
)


# ------------------------------------------------------------------ 1. Dataset
def tab_dataset():
    st.subheader("Ringkasan Dataset")
    n_tr, n_va, n_te = info.get("n_train", 0), info.get("n_val", 0), info.get("n_test", 0)
    counts = info.get("counts", {})
    c = st.columns(5)
    c[0].metric("Jumlah kelas", len(class_names))
    c[1].metric("Total gambar", "{:,}".format(info.get("total", 0)))
    c[2].metric("Data latih", "{:,}".format(n_tr))
    c[3].metric("Data validasi", "{:,}".format(n_va))
    c[4].metric("Data uji", "{:,}".format(n_te))
    tot = n_tr + n_va + n_te
    if tot:
        st.caption("Proporsi: latih {:.0f}% | validasi {:.0f}% | uji {:.0f}% "
                   "(80% data awal untuk latih+validasi, 20% untuk uji). Ukuran citra {}x{} grayscale.".format(
                       n_tr / tot * 100, n_va / tot * 100, n_te / tot * 100,
                       info["img_size"], info["img_size"]))

    if not counts:
        st.info("Data jumlah per kelas belum tersedia.")
    else:
        jumlah = [counts.get(k, 0) for k in class_names]
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Distribusi data per kelas**")
            fig, ax = plt.subplots(figsize=(7, 3.8))
            bars = ax.bar(class_names, jumlah, color="#4C72B0")
            ax.bar_label(bars, fontsize=8)
            ax.set_ylabel("Jumlah gambar")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with right:
            st.markdown("**Tabel jumlah gambar**")
            df = pd.DataFrame({"Emosi": class_names, "Jumlah": jumlah})
            df["Persen (%)"] = (df["Jumlah"] / df["Jumlah"].sum() * 100).round(2)
            st.dataframe(df)
            if jumlah and min(jumlah) > 0:
                rasio = max(jumlah) / min(jumlah)
                st.write("Rasio kelas terbanyak : tersedikit = {:.1f}x".format(rasio))
                if rasio > 1.5:
                    st.warning("Jumlah gambar antar kelas tidak seimbang; ditangani dengan class weight dan augmentasi.")

    st.markdown("**Contoh citra tiap kelas**")
    show_file("eda_contoh.png", width=700)


# ------------------------------------------------------------------ 2. Pelatihan
def tab_pelatihan():
    st.subheader("Hasil Pelatihan Model")
    if hist is None:
        st.info("history.json belum ada.")
        return
    ep = list(range(1, len(hist["loss"]) + 1))
    best = tinfo["best_epoch"] if tinfo else None
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, hist["accuracy"], label="Latih")
        ax.plot(ep, hist["val_accuracy"], label="Validasi")
        if best:
            ax.axvline(best, ls="--", c="gray", label="Terbaik (epoch {})".format(best))
        ax.set_title("Akurasi"); ax.set_xlabel("Epoch"); ax.set_ylabel("Akurasi"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ep, hist["loss"], label="Latih")
        ax.plot(ep, hist["val_loss"], label="Validasi")
        if best:
            ax.axvline(best, ls="--", c="gray", label="Terbaik (epoch {})".format(best))
        ax.set_title("Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    if tinfo:
        c = st.columns(4)
        c[0].metric("Epoch dijalani", "{} / {}".format(tinfo["epochs_dilatih"], tinfo["epochs_maks"]))
        c[1].metric("Epoch terbaik", tinfo["best_epoch"])
        c[2].metric("Total parameter", "{:,}".format(tinfo["total_param"]))
        c[3].metric("Durasi (menit)", tinfo["durasi_menit"])
        st.write("Optimizer: {} (lr awal {}) | Loss: {} | Batch size: {}".format(
            tinfo["optimizer"], tinfo["learning_rate"], tinfo["loss"], tinfo["batch_size"]))
        with st.expander("Lihat arsitektur CNN"):
            st.dataframe(pd.DataFrame(tinfo["layers"]))


# ------------------------------------------------------------------ 3. Evaluasi
def tab_evaluasi():
    st.subheader("Evaluasi Model pada Data Uji")
    if met is None:
        st.info("metrics.json belum ada.")
        return
    mode = st.radio("Rata-rata metrik", ["macro", "weighted"], horizontal=True)
    c = st.columns(4)
    c[0].metric("Accuracy", "{:.2f}%".format(met["accuracy"] * 100))
    c[1].metric("Precision", "{:.2f}%".format(met[mode]["precision"] * 100))
    c[2].metric("Recall", "{:.2f}%".format(met[mode]["recall"] * 100))
    c[3].metric("F1-score", "{:.2f}%".format(met[mode]["f1"] * 100))
    st.caption("Macro = rata-rata sederhana antar kelas; weighted = dibobot jumlah data tiap kelas.")

    dfp = pd.DataFrame(met["per_class"]).T[["precision", "recall", "f1", "support"]]
    dfp.columns = ["Precision", "Recall", "F1-score", "Support"]
    dfp["Support"] = dfp["Support"].astype(int)

    l, r = st.columns([2, 3])
    with l:
        st.markdown("**Metrik per kelas**")
        st.dataframe(dfp.round(3))
        terbaik, tersulit = dfp["F1-score"].idxmax(), dfp["F1-score"].idxmin()
        st.success("Paling mudah dikenali: {} (F1 {:.3f})".format(terbaik, dfp.loc[terbaik, "F1-score"]))
        st.error("Paling sulit dikenali: {} (F1 {:.3f})".format(tersulit, dfp.loc[tersulit, "F1-score"]))
    with r:
        st.markdown("**Confusion matrix**")
        norm = st.checkbox("Tampilkan ternormalisasi per baris", value=True)
        if cm is not None:
            data = cm / cm.sum(axis=1, keepdims=True) if norm else cm
            fig, ax = plt.subplots(figsize=(6.5, 5.2))
            im = heatmap(ax, data, class_names, ".2f" if norm else "d")
            fig.colorbar(im, ax=ax)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown("**Perbandingan F1-score antar kelas**")
    st.bar_chart(dfp["F1-score"])


# ------------------------------------------------------------------ 4. Visualisasi
def tab_visualisasi():
    st.subheader("Visualisasi Tambahan")
    st.markdown("**Grad-CAM** - area wajah yang paling memengaruhi keputusan model (merah = pengaruh tinggi)")
    show_file("gradcam.png")
    st.markdown("**Contoh prediksi vs label asli** (hijau = benar, merah = salah)")
    show_file("contoh_prediksi.png")
    st.markdown("**Contoh augmentasi data**")
    show_file("augmentasi_contoh.png", width=700)


# ------------------------------------------------------------------ 5. Prediksi
def tab_prediksi():
    st.subheader("Prediksi Emosi dari Gambar Baru")
    st.caption("Gunakan foto yang fokus pada satu wajah (close-up, menghadap depan). "
               "Citra otomatis diubah menjadi grayscale 48x48 sebelum diprediksi.")
    if not MODEL_PATH.exists():
        st.error("Model belum ditemukan di " + str(MODEL_PATH) + ". Jalankan notebook sampai Langkah 4.")
        return
    up = st.file_uploader("Unggah gambar (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if up is None:
        return

    img = ImageOps.exif_transpose(Image.open(up))
    small = img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(small, dtype="float32") / 255.0
    x = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    with st.spinner("Memuat model dan memprediksi..."):
        model = load_model()
        probs = model.predict(x, verbose=0)[0]
        last_conv = (tinfo or {}).get("last_conv", "relu3_2")
        try:
            cam = gradcam(x, last_conv)
        except Exception:
            cam = None

    k = int(np.argmax(probs))
    label = class_names[k]
    conf = float(probs[k]) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Gambar asli**")
        st.image(img, width=260)
    with c2:
        st.markdown("**Input model (48x48 grayscale)**")
        st.image(small.resize((260, 260), Image.NEAREST))
    with c3:
        if cam is not None:
            st.markdown("**Grad-CAM**")
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
            ax.imshow(arr, cmap="gray")
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.axis("off")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.success("Prediksi emosi: {} (keyakinan {:.1f}%)".format(label.upper(), conf))
    if conf < 50:
        st.warning("Keyakinan model rendah. Coba foto wajah yang lebih jelas dan close-up.")
    st.markdown("**Probabilitas tiap kelas (%)**")
    st.bar_chart(pd.Series(probs * 100, index=class_names))


# ------------------------------------------------------------------ jalankan tiap tab dengan penangkap error
def aman(fungsi):
    try:
        fungsi()
    except Exception as e:
        st.error("Terjadi error di bagian ini: {}".format(e))
        st.code(traceback.format_exc())


with tab1:
    aman(tab_dataset)
with tab2:
    aman(tab_pelatihan)
with tab3:
    aman(tab_evaluasi)
with tab4:
    aman(tab_visualisasi)
with tab5:
    aman(tab_prediksi)

st.divider()
st.caption("Praktikum Analisis Big Data - Klasifikasi Citra Emosi Wajah dengan CNN")