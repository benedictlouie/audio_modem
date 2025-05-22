import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from utils import *
from encoder import encode, get_known_blocks, get_chirp

def decode(signal: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Decode the received signal into symbols.
    """
    fourier = np.fft.fft(signal[:, CYCLIC_PREFIX:], axis=1) * filter
    symbols = fourier[:, 1:SYMBOLS_PER_BLOCK + 1].flatten()
    return symbols

def synchronize(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronize the received signal and return the filter.
    """
    chirp_signal = get_chirp()
    startIndex = np.argmax(np.correlate(signal, chirp_signal[::-1]))
    endIndex = np.argmax(np.correlate(signal, chirp_signal))
    
    known_blocks = get_known_blocks()

    frameLength = 2 * BLOCK_LENGTH
    left_bound = startIndex + CHIRP_LENGTH - frameLength//2

    received_known_blocks = np.empty((0, BLOCK_LENGTH))
    received_information_blocks = np.empty((0, BLOCK_LENGTH))

    sync_indices = []
    current_index = 0
    while left_bound + frameLength < endIndex:
        index = int(np.argmax(np.correlate(signal[left_bound:left_bound + frameLength], known_blocks[current_index]))) + left_bound
        sync_indices.append(index)
        left_bound = index + frameLength // 2
        
        received_known_blocks = np.vstack((received_known_blocks, signal[index:index + BLOCK_LENGTH]))
        received_information_blocks = np.vstack((received_information_blocks, signal[index + BLOCK_LENGTH:index + 2*BLOCK_LENGTH]))

        current_index += 1
    sync_diff = np.diff(sync_indices)
    print(f'Frame Length: {frameLength}, Mean Sync Diff: {np.mean(sync_diff):.2f}, Std Sync Diff: {np.std(sync_diff):.2f}')
    
    filter = estimate_filter(known_blocks, received_known_blocks, WIENER_SNR)
    return received_information_blocks, filter

def estimate_filter(sent_blocks: np.ndarray, receive_blocks: np.ndarray, snr: float) -> np.ndarray:
    """
    Remove the channel effect from the received signal using the sent signal.
    """
    sent = sent_blocks[:, CYCLIC_PREFIX:]
    received = receive_blocks[:, CYCLIC_PREFIX:]

    S = np.fft.fft(sent, axis=1)
    R = np.fft.fft(received, axis=1)
    H = R / (S + 1e-10)
    filter = np.conjugate(H) / (np.abs(H) ** 2 + 1 / snr)
    return filter

if __name__ == "__main__":
    AUDIO_PATH = "received.wav"
    signal = load_audio_file(AUDIO_PATH)
    signal, filter = synchronize(signal)
    received_symbols = decode(signal, filter)

    sent_symbols = get_symbols_from_bitstream(DATA)

    received_symbols = received_symbols[:len(sent_symbols)]
    if len(sent_symbols) != len(received_symbols): exit('synchronization error')
    received_symbols *= np.sqrt(2) / np.mean(np.abs(received_symbols))
    plot_sent_received_constellation(sent_symbols, received_symbols)

    received_data = get_bitstream_from_symbols(received_symbols)[:len(DATA)]
    print(f'Error Rate: {np.sum(received_data != DATA) / len(DATA) * 100:.2f}%')