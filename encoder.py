import numpy as np

from utils import *

def encode(symbols: np.ndarray) -> np.ndarray:
    """
    Encode a bitstream into a time-domain signal using IFFT and add a cyclic prefix.
    """
    symbols = np.concatenate((symbols, np.repeat(1 + 1j, (-len(symbols)) % SYMBOLS_PER_BLOCK))).reshape(-1, SYMBOLS_PER_BLOCK)
    encoded_symbols = np.concatenate((np.zeros((symbols.shape[0], 1)), symbols, np.zeros((symbols.shape[0], 1)), np.conjugate(symbols[:, ::-1])), axis=1)
    symbolsInTime = np.fft.ifft(encoded_symbols, axis=1).real
    signal = np.concatenate((symbolsInTime[:, -CYCLIC_PREFIX:], symbolsInTime), axis=1).flatten()
    return signal / np.max(np.abs(signal))

def insert_known_blocks(signal: np.ndarray) -> np.ndarray:
    """
    Insert a synchronization chirp before each block of the signal.
    """
    known_blocks = get_known_blocks(len(signal) // BLOCK_LENGTH)
    blocks = signal.reshape(-1, BLOCK_LENGTH)
    return np.concatenate((known_blocks, blocks), axis=1).flatten()

def get_known_blocks(blockCount: int) -> np.ndarray:
    """
    Generate known blocks of symbols for synchronization.
    """
    symbols = get_symbols_from_bitstream(get_non_repeating_bits(BITS_PER_CONSTELLATION * SYMBOLS_PER_BLOCK * blockCount, 1), skip_encoding=True)
    return encode(symbols).reshape(-1, BLOCK_LENGTH)

def insert_chirps(signal: np.ndarray) -> np.ndarray:
    """
    Insert pilot signals at the beginning and end of the signal.
    """
    chirp = get_chirp()
    return np.concatenate((chirp[::-1], signal, chirp))

def get_chirp() -> np.ndarray:
    """
    Generate a pilot signal using a chirp signal.
    """
    t = np.linspace(0, CHIRP_TIME, CHIRP_LENGTH)
    signal = CHIRP_FACTOR * np.sin(2 * np.pi * (CHIRP_LOW + (CHIRP_HIGH - CHIRP_LOW) * t / CHIRP_TIME) * t)
    return signal
    
if __name__ == "__main__":
    symbols = get_symbols_from_bitstream(DATA)
    signal = encode(symbols)
    signal = insert_known_blocks(signal)
    signal = insert_chirps(signal)

    print(f'Bitrate: {round(len(DATA) * SAMPLE_RATE / len(signal))} bps')
    print(f'Bitstream: {len(DATA)} bits')
    print(f'Time: {(len(signal) / SAMPLE_RATE):.2f} seconds')

    signal = np.concatenate((np.zeros(SAMPLE_RATE), signal, np.zeros(SAMPLE_RATE)))
    write_wav(AUDIO_PATH, signal)