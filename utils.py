import numpy as np
import matplotlib.pyplot as plt
import ldpc
import librosa
from scipy.io.wavfile import write
from scipy.signal import correlate

# All constants

FULL_SYMBOLS_PER_BLOCK = 4095
N_DFT = 2 * (FULL_SYMBOLS_PER_BLOCK + 1)
CYCLIC_PREFIX = N_DFT // 4
BLOCK_LENGTH = N_DFT + CYCLIC_PREFIX

SAMPLING_RATE = 48000

CHIRP_FACTOR = 0.01
CHIRP_LENGTH = BLOCK_LENGTH
CHIRP_LOW = 20
CHIRP_HIGH = 20000

NUMBER_OF_PILOT_BLOCKS = 8

LOW_PASS_INDEX = round(0.9 * FULL_SYMBOLS_PER_BLOCK)
HIGH_PASS_INDEX = round(0.01 * FULL_SYMBOLS_PER_BLOCK)
SYMBOLS_PER_BLOCK = LOW_PASS_INDEX - HIGH_PASS_INDEX

DEFAULT_AUDIO_PATH = "untitled.wav"

# Define QPSK mapping and modulation
MAPPING = {
    (0, 0): 1 + 1j,
    (0, 1): 1 - 1j,
    (1, 0): -1 + 1j,
    (1, 1): -1 - 1j,
}
MAPPING = {k: v / np.sqrt(2) for k, v in MAPPING.items()}
INV_MAPPING = {v: k for k, v in MAPPING.items()}
BITS_PER_SYMBOL = 2

# WIENER FILTER
SNR = 10

# LDPC SETTINGS
DECTYPE = 'sumprod2'
CODE = ldpc.code(z=81)

def generate_data_bits(num_blocks):
    num_bits = BITS_PER_SYMBOL * SYMBOLS_PER_BLOCK * num_blocks
    bits = np.random.default_rng(42).integers(0, 2, num_bits)
    return bits

def generate_pilot_bits():
    pilot_bits = np.random.default_rng(69).integers(0, 2, BITS_PER_SYMBOL * SYMBOLS_PER_BLOCK * NUMBER_OF_PILOT_BLOCKS)
    return pilot_bits

def encodeLDPC(bits):
    bits = np.concatenate((bits, np.zeros((-len(bits)) % CODE.K)))
    encoded_bitstream = np.array([])
    for i in range(0, len(bits), CODE.K):
        encoded_bitstream = np.concatenate((encoded_bitstream, CODE.encode(bits[i:i + CODE.K])))
    return encoded_bitstream

def modulate(bits):
    bits = bits.reshape((-1, BITS_PER_SYMBOL))
    symbols = np.array([MAPPING[tuple(b)] for b in bits])
    return symbols

def bitsToSymbols(bits):
    encoded_bitstream = encodeLDPC(bits)
    symbols = modulate(encoded_bitstream)
    post_padding = modulate(np.random.randint(2, size=(-len(symbols)) % SYMBOLS_PER_BLOCK * BITS_PER_SYMBOL))
    symbols = np.concatenate((symbols, post_padding))
    return symbols

def generate_chirp(length, fs, f0, f1):
    t = np.linspace(0, length / fs, length, endpoint=False)
    k = (f1 - f0) / (length / fs)
    return CHIRP_FACTOR * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))

bits = generate_data_bits(20)