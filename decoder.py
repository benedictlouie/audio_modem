import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from utils import *
from encoder import encode, get_sync_signal, get_start_end_signal

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
    start_end_signal = get_start_end_signal()
    startIndex = np.argmax(np.correlate(signal, start_end_signal))
    endIndex = np.argmax(np.correlate(signal, start_end_signal[::-1]))
    
    sync_signal = get_sync_signal()
    sync_correlate = np.correlate(signal, sync_signal)

    totalLength = SYNC_CHIRP_LENGTH + BLOCK_LENGTH * BLOCKS_PER_SYNC
    left_bound = startIndex + START_END_CHIRP_LENGTH - totalLength//2
    received_blocks = np.empty((0, BLOCK_LENGTH))

    sync_indices = []
    while left_bound + totalLength < endIndex:
        index = int(np.argmax(sync_correlate[left_bound:left_bound + totalLength]) + left_bound)
        sync_indices.append(index)
        left_bound = index + totalLength // 2
        received_blocks = np.vstack((received_blocks, signal[index + SYNC_CHIRP_LENGTH:index + totalLength].reshape(-1, BLOCK_LENGTH)))
    sync_diff = np.diff(sync_indices)
    print(f'Mean Sync Diff: {np.mean(sync_diff):.2f}, Std Sync Diff: {np.std(sync_diff):.2f}')
    
    sent_channel_estimate_blocks = encode(get_symbols_from_bitstream(ESTIMATE_CHANNEL_DATA, skip_encoding=True)).reshape(-1, BLOCK_LENGTH)
    received_channel_estimate_blocks = received_blocks[:ESTIMATE_CHANNEL_BLOCKS, :]
    filter = estimate_filter(sent_channel_estimate_blocks, received_channel_estimate_blocks, WIENER_SNR)
    return received_blocks[ESTIMATE_CHANNEL_BLOCKS:, :], filter

def estimate_filter(sent_blocks: np.ndarray, receive_blocks: np.ndarray, snr: float) -> np.ndarray:
    """
    Remove the channel effect from the received signal using the sent signal.
    """
    # filterLength = BLOCK_LENGTH - CYCLIC_PREFIX
    # sent = np.empty((0, filterLength))
    # received = np.empty((0, filterLength))
    # for i in range(CYCLIC_PREFIX):
    #     sent = np.vstack((sent, sent_blocks[:, i:i + filterLength]))
    #     received = np.vstack((received, receive_blocks[:, i:i + filterLength]))
    
    sent = sent_blocks[:, CYCLIC_PREFIX:]
    received = receive_blocks[:, CYCLIC_PREFIX:]

    S = np.fft.fft(sent, axis=1)
    R = np.fft.fft(received, axis=1)
    H = R / (S + 1e-10)
    filter = np.conjugate(H) / (np.abs(H) ** 2 + 1 / snr)
    filter = np.mean(filter, axis=0)
    return filter

if __name__ == "__main__":
    AUDIO_PATH = "Downing College.m4a"
    signal = load_audio_file(AUDIO_PATH)
    signal, filter = synchronize(signal)

    sent_symbols = get_symbols_from_bitstream(DATA)
    received_symbols = decode(signal, filter)
    received_symbols = received_symbols[:len(sent_symbols)]
    received_symbols *= np.sqrt(2) / np.mean(np.abs(received_symbols))
    

    plot_sent_received_constellation(sent_symbols, received_symbols)

    received_data = get_bitstream_from_symbols(received_symbols)[:len(DATA)]
    print(f'Error Rate: {np.sum(received_data != DATA) / len(DATA) * 100:.2f}%')