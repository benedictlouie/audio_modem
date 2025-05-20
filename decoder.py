import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from utils import *
from encoder import get_pilot_signal, get_sync_chirp

def decode_block(data: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Decode a block of data using FFT and extract the symbols.
    """
    fourier = np.fft.fft(data[CYCLIC_PREFIX:]) * filter
    return fourier[1:SYMBOLS_PER_BLOCK + 1]

def decode(signal: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Decode the received signal into a bitstream.
    """

    blockCount = len(signal) // BLOCK_LENGTH
    symbols = np.zeros(blockCount * SYMBOLS_PER_BLOCK, dtype=complex)
    for i in range(blockCount):
        symbols[i * SYMBOLS_PER_BLOCK: (i+1) * SYMBOLS_PER_BLOCK] = decode_block(signal[i * BLOCK_LENGTH: (i+1) * BLOCK_LENGTH], filter)
    return symbols

def synchronize(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronize the received signal and return the filter.
    """
    pilot = get_pilot_signal()
    startIndex = np.argmax(np.correlate(signal, pilot))
    endIndex = np.argmax(np.correlate(signal, pilot[::-1]))
    
    sync_chirp = get_sync_chirp()
    sync_correlate = np.correlate(signal, sync_chirp)

    left_bound = startIndex + CHIRP_LENGTH - BLOCK_LENGTH
    output_blocks = []
    sent_blocks = []
    received_blocks = []
    while left_bound + 2 * BLOCK_LENGTH < endIndex:
        index = int(np.argmax(sync_correlate[left_bound:left_bound + 2 * BLOCK_LENGTH]) + left_bound)
        left_bound = index + BLOCK_LENGTH
        output_blocks.append(signal[index + BLOCK_LENGTH:index + 2 * BLOCK_LENGTH])

        sent_block = np.array([sync_chirp[i:BLOCK_LENGTH - CYCLIC_PREFIX + i] for i in range(CYCLIC_PREFIX)])
        received_block = np.array([signal[index + i:index + BLOCK_LENGTH - CYCLIC_PREFIX + i] for i in range(CYCLIC_PREFIX)])
        sent_blocks.append(sent_block)
        received_blocks.append(received_block)
    output = np.concatenate(output_blocks)
    sent = np.concatenate(sent_blocks)
    received = np.concatenate(received_blocks)
    
    filter = estimate_filter(sent, received, 0.1)
    return output, filter

def estimate_filter(sent: np.ndarray, received: np.ndarray, snr: float) -> np.ndarray:
    """
    Remove the channel effect from the received signal using the sent signal.
    """
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
    received_symbols = decode(signal, filter)
    sent_symbols = get_symbols_from_bitstream(DATA)

    plot_sent_received_constellation(sent_symbols, received_symbols[:len(sent_symbols)])

    received_data = get_bitstream_from_symbols(received_symbols)[:len(DATA)]
    print(f'Error Rate: {np.sum(received_data != DATA) / len(DATA) * 100:.2f}%')