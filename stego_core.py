import numpy as np
import soundfile as sf

ALPHA = 0.01
SAMPLES_PER_BIT = 400
SEED = 42

# -------- TEXT ↔ BITS --------
def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8))

# -------- AUDIO --------
def read_audio(file_path):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio / np.max(np.abs(audio))
    return audio, sr

# -------- PN --------
def generate_pn(length):
    np.random.seed(SEED)
    return np.random.choice([-1, 1], length)

# -------- EMBED --------
def embed_message(audio, message):
    bits = text_to_bits(message)
    pn = generate_pn(SAMPLES_PER_BIT)

    stego = np.copy(audio)
    index = 0
    for bit in bits:
        spread = pn if bit == '1' else -pn
        stego[index:index + SAMPLES_PER_BIT] += ALPHA * spread
        index += SAMPLES_PER_BIT

    return stego, len(bits)

# -------- EXTRACT --------
def extract_message(stego, original, total_bits):
    pn = generate_pn(SAMPLES_PER_BIT)
    bits = ""
    index = 0

    for _ in range(total_bits):
        diff = stego[index:index + SAMPLES_PER_BIT] - original[index:index + SAMPLES_PER_BIT]
        bits += '1' if np.dot(diff, pn) > 0 else '0'
        index += SAMPLES_PER_BIT

    return bits_to_text(bits)

# -------- METRICS --------
def calculate_snr(original, stego):
    noise = original - stego
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    return float('inf') if noise_power == 0 else 10 * np.log10(signal_power / noise_power)

def calculate_psnr(original, stego):
    mse = np.mean((original - stego) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(np.abs(original))
    return 10 * np.log10((max_val ** 2) / mse)

def calculate_accuracy(original_msg, extracted_msg):
    orig_bits = text_to_bits(original_msg)
    ext_bits = text_to_bits(extracted_msg)

    min_len = min(len(orig_bits), len(ext_bits))
    correct = sum(1 for i in range(min_len) if orig_bits[i] == ext_bits[i])

    return (correct / min_len) * 100
