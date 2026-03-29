import streamlit as st
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt

from stego_core import (
    read_audio, embed_message, extract_message,
    calculate_snr, calculate_psnr, calculate_accuracy
)
from predict import predict_audio

st.set_page_config(page_title="Audio Steganography", layout="centered")

st.title("🔐 Secure Data Transmission")
st.subheader("Audio Steganography using Spread Spectrum")

uploaded_file = st.file_uploader("Upload WAV Audio", type=["wav"])
secret_message = st.text_input("Enter Secret Message")

if "data" not in st.session_state:
    st.session_state.data = {}

# -------- EMBED --------
if st.button("Embed Message"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    original, sr = read_audio(path)
    stego, bits = embed_message(original, secret_message)

    snr = calculate_snr(original, stego)
    psnr = calculate_psnr(original, stego)

    sf.write("stego.wav", stego, sr)

    st.session_state.data = {
        "original": original,
        "stego": stego,
        "bits": bits,
        "message": secret_message
    }

    st.success("✅ Embedding Successful")
    st.metric("SNR (dB)", f"{snr:.2f}")
    st.metric("PSNR (dB)", f"{psnr:.2f}")

    with open("stego.wav", "rb") as f:
        st.download_button("Download Stego Audio", f, "stego.wav")

# -------- EXTRACT & ACCURACY --------
if st.button("Extract Message"):
    d = st.session_state.data

    extracted = extract_message(d["stego"], d["original"], d["bits"])
    accuracy = calculate_accuracy(d["message"], extracted)

    st.subheader("📊 Comparison Results")
    st.success(f"Extracted Message: {extracted}")
    st.metric("Bit Accuracy (%)", f"{accuracy:.2f}")

    if extracted == d["message"]:
        st.success("✅ Messages MATCH")
    else:
        st.error("❌ Messages DO NOT Match")

    # Waveform comparison
    st.subheader("📈 Audio Comparison")
    fig, ax = plt.subplots()
    ax.plot(d["original"][:2000], label="Original")
    ax.plot(d["stego"][:2000], linestyle="dashed", label="Stego")
    ax.legend()
    st.pyplot(fig)

# -------- ML DETECTION --------
st.subheader("🧠 CNN Steganalysis")
test_audio = st.file_uploader("Upload Audio for Detection", type=["wav"], key="ml")

if st.button("Analyze Audio"):
    with open("test.wav", "wb") as f:
        f.write(test_audio.read())
    result = predict_audio("test.wav")
    st.success(f"Result: {result}")
