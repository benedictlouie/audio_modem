import numpy as np
from typing import Tuple

import matplotlib.pyplot as plt

from utils import *

def encode_block(symbols: np.ndarray) -> np.ndarray:
    """
    Encode a block of symbols into a time-domain signal using IFFT and add a cyclic prefix.
    """
    symbols = np.concatenate((np.zeros(1), symbols, np.zeros(1), np.conjugate(symbols[::-1])))
    symbolsInTime = np.fft.ifft(symbols).real
    return np.concatenate((symbolsInTime[-CYCLIC_PREFIX:], symbolsInTime))

def encode(symbols: np.ndarray) -> np.ndarray:
    """
    Encode a bitstream into a time-domain signal using IFFT and add a cyclic prefix.
    """
    blockCount = len(symbols) // SYMBOLS_PER_BLOCK
    signal = np.zeros(blockCount * BLOCK_LENGTH)
    for i in range(blockCount):
        signal[i * BLOCK_LENGTH : (i+1) * BLOCK_LENGTH] = encode_block(symbols[i * SYMBOLS_PER_BLOCK : (i+1) * SYMBOLS_PER_BLOCK])
    return signal

def insert_pilot_signals(signal: np.ndarray) -> np.ndarray:
    """
    Insert pilot signals at the beginning and end of the signal.
    """
    pilot = get_pilot_signal()
    return np.concatenate((pilot, signal, pilot[::-1]))

def get_pilot_signal() -> np.ndarray:
    """
    Generate a pilot signal using a chirp signal.
    """
    t = np.linspace(0, CHIRP_TIME, CHIRP_LENGTH)
    chirp = CHIRP_FACTOR * np.sin(2 * np.pi * (CHIRP_LOW + (CHIRP_HIGH - CHIRP_LOW) * t / CHIRP_TIME) * t)
    return chirp

def get_filter_blocks() -> np.ndarray:
    """
    Generate filter blocks for the signal.
    """
    return FILTER_MULTIPLIER * get_white_noise(FILTER_BLOCKS * BLOCK_LENGTH, seed=0)
    
if __name__ == "__main__":
    symbols = get_symbols_from_bitstream(DATA)
    signal = encode(symbols)
    signal = np.concatenate((get_filter_blocks(), signal))
    signal = insert_pilot_signals(signal)
    signal = np.concatenate((np.zeros(SAMPLE_RATE), signal, np.zeros(SAMPLE_RATE)))
    write_wav(AUDIO_PATH, signal)

    print(f'Bitrate: {round(len(DATA) * SAMPLE_RATE / (len(signal) - 2 * SAMPLE_RATE))} bps')
    print(f'Bitstream: {len(DATA)} bits')
    print(f'Time: {(len(signal) / SAMPLE_RATE - 2):.2f} seconds')