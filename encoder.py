import numpy as np

from utils import *

def encode(symbols: np.ndarray) -> np.ndarray:
    """
    Encode a bitstream into a time-domain signal using IFFT and add a cyclic prefix.
    """
    symbols = np.concatenate((symbols, np.repeat(1 + 1j, (-len(symbols)) % (SYMBOLS_PER_BLOCK * BLOCKS_PER_SYNC)))).reshape(-1, SYMBOLS_PER_BLOCK)
    encoded_symbols = np.concatenate((np.zeros((symbols.shape[0], 1)), symbols, np.zeros((symbols.shape[0], 1)), np.conjugate(symbols[:, ::-1])), axis=1)
    symbolsInTime = np.fft.ifft(encoded_symbols, axis=1).real
    signal = np.concatenate((symbolsInTime[:, -CYCLIC_PREFIX:], symbolsInTime), axis=1).flatten()
    return signal / np.max(np.abs(signal))

def insert_sync_signal(signal: np.ndarray) -> np.ndarray:
    """
    Insert a synchronization chirp before each block of the signal.
    """
    sync_signal = get_sync_signal()
    blocks = signal.reshape(-1, BLOCK_LENGTH * BLOCKS_PER_SYNC)
    blocks_with_chirp = np.zeros((blocks.shape[0], SYNC_CHIRP_LENGTH + BLOCK_LENGTH * BLOCKS_PER_SYNC))
    blocks_with_chirp[:, :SYNC_CHIRP_LENGTH] = sync_signal
    blocks_with_chirp[:, SYNC_CHIRP_LENGTH:] = blocks
    return blocks_with_chirp.flatten()

def insert_start_end_signal(signal: np.ndarray) -> np.ndarray:
    """
    Insert pilot signals at the beginning and end of the signal.
    """
    tmp = get_start_end_signal()
    return np.concatenate((tmp, signal, tmp[::-1]))

def get_sync_signal() -> np.ndarray:
    """
    Generate a synchronization chirp signal.
    """
    t = np.linspace(0, SYNC_CHIRP_TIME, SYNC_CHIRP_LENGTH)
    chirp = CHIRP_FACTOR * np.sin(2 * np.pi * (CHIRP_LOW + (CHIRP_HIGH - CHIRP_LOW) * t/ SYNC_CHIRP_TIME) * t)
    return chirp

def get_start_end_signal() -> np.ndarray:
    """
    Generate a pilot signal using a chirp signal.
    """
    t = np.linspace(0, START_END_CHIRP_TIME, START_END_CHIRP_LENGTH)
    signal = CHIRP_FACTOR * np.sin(2 * np.pi * (CHIRP_LOW + (CHIRP_HIGH - CHIRP_LOW) * t / START_END_CHIRP_TIME) * t)
    return signal
    
if __name__ == "__main__":
    symbols = np.concatenate((get_symbols_from_bitstream(ESTIMATE_CHANNEL_DATA, skip_encoding=True), get_symbols_from_bitstream(DATA)))
    signal = encode(symbols)
    signal = insert_sync_signal(signal)
    signal = insert_start_end_signal(signal)

    print(f'Bitrate: {round(len(DATA) * SAMPLE_RATE / len(signal))} bps')
    print(f'Bitstream: {len(DATA)} bits')
    print(f'Time: {(len(signal) / SAMPLE_RATE):.2f} seconds')

    signal = np.concatenate((np.zeros(SAMPLE_RATE), signal, np.zeros(SAMPLE_RATE)))
    write_wav(AUDIO_PATH, signal)