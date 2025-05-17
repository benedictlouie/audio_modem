import contextlib
import librosa
import numpy as np
from scipy.io.wavfile import write

# Number of Fourier Symbols every DFT block
symbolsPerBlock = 200
cyclicPrefix = symbolsPerBlock

# +1 for the zeroth bit and x2 for conjugate
# If symbolsPerBlock = 511 and cyclicPrefix = 32, blockLength = 1056
blockLength = 2 * (symbolsPerBlock + 1) + cyclicPrefix

# Sample rate of audio
sampleRate = 48000

# How many times we repeat the symbols
repeatCount = symbolsPerBlock

# Sync block has length blockLength
# It happens every syncBlockPeriod blocks
syncBlockPeriod = 20
startEndBlockMultiplier = 2 # TODO: this doesn't work at higher values, not sure why -G
syncLength = syncBlockPeriod * blockLength

# Constellation
constellation = {
    '00': 1 + 1j,
    '01': -1 + 1j,
    '10': 1 - 1j,
    '11': -1 - 1j
}

audio_path = "output.wav"

def load_audio_file(file_path: str) -> np.ndarray:
    with contextlib.redirect_stderr(None):
        return librosa.load(file_path, sr=None)[0]

def write_wav(filename: str, data: np.ndarray, sample_rate: int = sampleRate) -> None:
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def get_non_repeating_bits(n: int) -> str:
    # Generates the same binary numbers every time
    rng = np.random.default_rng(seed=42)
    bits = rng.integers(0, 2, size=n)
    return ''.join(map(str, bits))

def text_to_binary(text: str) -> str:
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_str: str) -> str:
    return ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))

def encode_ldpc(bits: str) -> str:
    bits += "0" * ((-len(bits)) % 4)

    data_bits = list(map(int, bits))
    encoded = []

    # positions (1-indexed): 1-p1, 2-p2, 3-d1, 4-p3, 5-d2, 6-d3, 7-d4
    for i in range(0, len(data_bits), 4):
        d1, d2, d3, d4 = data_bits[i : i + 4]

        p1 = d1 ^ d2 ^ d4          # parity for positions 1,3,5,7
        p2 = d1 ^ d3 ^ d4          # parity for positions 2,3,6,7
        p3 = d2 ^ d3 ^ d4          # parity for positions 4,5,6,7

        encoded.extend([p1, p2, d1, p3, d2, d3, d4])

    return "".join(map(str, encoded))


def decode_ldpc(bits: str) -> str:
    # pad to a multiple of 7 coded bits
    bits += "0" * ((-len(bits)) % 7)

    coded_bits = list(map(int, bits))
    decoded = []

    for i in range(0, len(coded_bits), 7):
        block = coded_bits[i : i + 7]
        p1, p2, d1, p3, d2, d3, d4 = block

        # syndrome bits (s1 = LSB, s3 = MSB)
        s1 = p1 ^ d1 ^ d2 ^ d4
        s2 = p2 ^ d1 ^ d3 ^ d4
        s3 = p3 ^ d2 ^ d3 ^ d4
        error_pos = (s3 << 2) | (s2 << 1) | s1   # 0 = no error, 1-7 = bit position to flip

        if error_pos:                             # single-bit error detected
            block[error_pos - 1] ^= 1             # correct it
            p1, p2, d1, p3, d2, d3, d4 = block    # refresh variables after fix

        decoded.extend([d1, d2, d3, d4])

    return "".join(map(str, decoded))

    

    

