import numpy as np

from utils import *

def write_wav(filename: str, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def insert_known_blocks(signal: np.ndarray) -> np.ndarray:
    """
    Insert a synchronization OFDM block before every information block of the signal.
    """
    # With varicable frame length
    known_blocks = get_known_blocks()
    blocks = signal.reshape(-1, BLOCK_LENGTH * INFORMATION_BLOCKS_PER_FRAME)
    return np.concatenate((known_blocks, blocks), axis=1).flatten()

def insert_chirps(signal: np.ndarray) -> np.ndarray:
    """
    Insert a reverse chirp at the beginning and chirp at the end of the signal.
    """
    chirp = get_chirp()
    return np.concatenate((chirp[::-1], signal, chirp))
    
if __name__ == "__main__":
    symbols = get_symbols_from_bitstream(DATA)
    signal = encode(symbols)
    signal = insert_known_blocks(signal)
    signal = insert_chirps(signal)

    print(f'Bitrate: {round(len(DATA) * SAMPLE_RATE / len(signal))} bps')
    print(f'Bitstream: {len(DATA)} bits')
    print(f'Time: {(len(signal) / SAMPLE_RATE):.2f} seconds')

    # Insert one second of nothing before and after the signal
    signal = np.concatenate((np.zeros(SAMPLE_RATE), signal, np.zeros(SAMPLE_RATE)))
    write_wav(AUDIO_PATH, signal)