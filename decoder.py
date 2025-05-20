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

def get_filter(sent: np.ndarray, received: np.ndarray, snr: float) -> np.ndarray:
    """
    Calculate the filter for the channel using the sent and received signals.
    """
    S = np.fft.fft(sent)
    R = np.fft.fft(received)
    H = R / (S + 1e-10)
    return np.conjugate(H) / (np.abs(H) ** 2 + 1 / snr)

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
    output = np.array([])
    received = np.array([])
    while left_bound + 2*BLOCK_LENGTH < endIndex:
        index = int(np.argmax(sync_correlate[left_bound:left_bound + 2*BLOCK_LENGTH]) + left_bound)
        left_bound = index + BLOCK_LENGTH

        received = np.concatenate((received, signal[index:index + BLOCK_LENGTH]))
        output = np.concatenate((output, signal[index + BLOCK_LENGTH:index + 2*BLOCK_LENGTH]))

    noise_power = np.mean(signal[:startIndex] ** 2)
    signal_power = np.mean(output ** 2)
    snr = signal_power / noise_power
    print(f'noise power: {noise_power}, signal power: {signal_power}, SNR: {snr:.2f}')
    
    received = np.reshape(received, (-1, BLOCK_LENGTH))
    filter = estimate_filter(sync_chirp[CYCLIC_PREFIX:], received[:, CYCLIC_PREFIX:], snr)
    return output, filter

def estimate_filter(sent: np.ndarray, received: np.ndarray, snr: float) -> np.ndarray:
    """
    Remove the channel effect from the received signal using the sent signal.
    """
    filter = np.zeros(received.shape, dtype=complex)
    for i in range(received.shape[0]):
        filter[i] = get_filter(sent, received[i], snr)
    filter = np.mean(filter, axis=0)
    # h = np.fft.ifft(filter).real
    # plt.plot(h)
    # plt.show()
    return filter

if __name__ == "__main__":
    AUDIO_PATH = "Downing College.m4a"
    signal = load_audio_file(AUDIO_PATH)
    signal, filter = synchronize(signal)

    sent_symbols = get_symbols_from_bitstream(DATA)
    received_symbols = decode(signal, filter)
    if len(received_symbols) != len(sent_symbols): exit('synchronization failed')

    plot_sent_received_constellation(sent_symbols, received_symbols)

    received_data = get_bitstream_from_symbols(received_symbols)[:len(DATA)]
    print(f'Error Rate: {np.sum(np.array(list(received_data)) != np.array(list(DATA))) / len(DATA) * 100:.2f}%')
