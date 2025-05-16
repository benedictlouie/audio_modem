import numpy as np
from scipy.io.wavfile import write
import librosa

symbolsPerBlock = 1024
cyclicPrefix = 128
blockLength = 2 * (symbolsPerBlock + 1) + cyclicPrefix
sampleRate = 48000
symbolRate = 100
syncBlockPeriod = 10
audio_path = "output.wav"

def load_audio_file(file_path):
    return librosa.load(file_path, sr=None)[0]

def write_wav(filename, data, sample_rate=sampleRate):
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def get_non_repeating_bits(n):
    rng = np.random.default_rng(seed=42)
    bits = rng.integers(0, 2, size=n)
    return ''.join(map(str, bits))

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_str):
    return ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))