import numpy as np
from scipy.io.wavfile import write
import librosa

# Number of Fourier Symbols every DFT block
symbolsPerBlock = 511

cyclicPrefix = 32

# +1 for the zeroth bit and x2 for conjugate
# If symbolsPerBlock = 511 and cyclicPrefix = 32, blockLength = 1056
blockLength = 2 * (symbolsPerBlock + 1) + cyclicPrefix

# Sample rate of audio
sampleRate = 48000

# We transmit symbolRate symbols every second
# if sampleRate = 48000 and symbolRate = 100, we repeat each symbol 480 times
symbolRate = 100

# Sync block has length blockLength
# It happens every syncBlockPeriod blocks
syncBlockPeriod = 10

# Constellation
constellation = {
    '00': 1 + 1j,
    '01': -1 + 1j,
    '10': 1 - 1j,
    '11': -1 - 1j
}

audio_path = "output.wav"

def load_audio_file(file_path):
    return librosa.load(file_path, sr=None)[0]

def write_wav(filename, data, sample_rate=sampleRate):
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def get_non_repeating_bits(n):
    # Generates the same binary numbers every time
    rng = np.random.default_rng(seed=42)
    bits = rng.integers(0, 2, size=n)
    return ''.join(map(str, bits))

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_str):
    return ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))