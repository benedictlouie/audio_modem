import numpy as np
import matplotlib.pyplot as plt

from utils import *
from encoder import get_pilot_signal, get_sync_chirp

def decode_block(data: np.ndarray) -> np.ndarray:
    """
    Decode a block of data using FFT and extract the symbols.
    """
    fourier = np.fft.fft(data[CYCLIC_PREFIX:])[1:SYMBOLS_PER_BLOCK + 1]
    return fourier

def decode(signal: np.ndarray) -> np.ndarray:
    """
    Decode the received signal into a bitstream.
    """

    blockCount = len(signal) // BLOCK_LENGTH
    symbols = np.zeros(blockCount * SYMBOLS_PER_BLOCK, dtype=complex)
    for i in range(blockCount):
        symbols[i * SYMBOLS_PER_BLOCK: (i+1) * SYMBOLS_PER_BLOCK] = decode_block(signal[i * BLOCK_LENGTH: (i+1) * BLOCK_LENGTH])
    return symbols

def get_filter(sent: np.ndarray, received: np.ndarray, snr: float) -> np.ndarray:
    """
    Calculate the filter for the channel using the sent and received signals.
    """
    S = np.fft.fft(sent)
    R = np.fft.fft(received)
    H = R / (S + 1e-10)
    return np.conjugate(H) / (np.abs(H) ** 2 + 1 / snr)

def synchronize(signal: np.ndarray) -> np.ndarray:
    """
    Synchronize the received signal.
    """
    pilot = get_pilot_signal()
    startIndex = np.argmax(np.correlate(signal, pilot))
    endIndex = np.argmax(np.correlate(signal, pilot[::-1]))

    sent = np.concatenate((pilot, pilot[::-1]))
    received = np.concatenate((signal[startIndex:startIndex + CHIRP_LENGTH], signal[endIndex:endIndex + CHIRP_LENGTH]))

    signal = remove_channel(signal, sent, received, SNR, CHIRP_FILTER_DIVISOR)

    totalLength = BLOCK_LENGTH + SYNC_CHIRP_LENGTH
    signal = signal[startIndex + CHIRP_LENGTH - totalLength//2:endIndex]
    
    sync_chirp = get_sync_chirp()
    sync_correlate = np.correlate(signal, sync_chirp)
    left_bound = 0
    count = 0
    output = np.array([])
    received = np.array([])
    while left_bound + totalLength < len(signal):
        index = int(np.argmax(sync_correlate[left_bound:left_bound + totalLength]) + left_bound)
        left_bound = index + totalLength // 2

        count += 1
        received = np.concatenate((received, signal[index:index + SYNC_CHIRP_LENGTH]))
        output = np.concatenate((output, signal[index + SYNC_CHIRP_LENGTH:index + totalLength]))

    return output

def remove_channel(signal: np.ndarray, sent: np.ndarray, received: np.ndarray, snr: float, filterDivisor: int) -> np.ndarray:
    """
    Remove the channel effect from the received signal using the sent signal.
    """
    filterLength = len(sent) // filterDivisor
    filter = np.zeros(filterLength, dtype=complex)
    for i in range(0, len(sent) - filterLength + 1, filterLength):
        filter += get_filter(sent[i:i+filterLength], received[i:i+filterLength], snr) / filterDivisor
    h = np.fft.ifft(filter).real
    h = h[:filterLength // 2]
    plt.plot(h)
    plt.show()
    return np.convolve(signal, h, mode='same')

if __name__ == "__main__":
    AUDIO_PATH = "Downing College.m4a"
    signal = load_audio_file(AUDIO_PATH)
    signal = synchronize(signal)

    sent_symbols = get_symbols_from_bitstream(DATA)
    received_symbols = decode(signal)
    if len(received_symbols) != len(sent_symbols): exit('synchronization failed')

    plot_sent_received_constellation(sent_symbols, received_symbols)

    received_data = get_bitstream_from_symbols(received_symbols)[:len(DATA)]
    print(f'Error Rate: {np.sum(np.array(list(received_data)) != np.array(list(DATA))) / len(DATA) * 100:.2f}%')
