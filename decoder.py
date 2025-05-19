import numpy as np
import matplotlib.pyplot as plt

from utils import *
from encoder import get_pilot_signal, get_filter_blocks

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

    noise_power = np.mean(np.concatenate((signal[:startIndex], signal[endIndex + CHIRP_LENGTH:])) ** 2)
    signal = signal[startIndex + CHIRP_LENGTH:endIndex]

    targetLength = round(len(signal) / BLOCK_LENGTH) * BLOCK_LENGTH
    # linear interpolations
    # signal = np.interp(np.linspace(0, len(signal) - 1, targetLength), np.arange(len(signal)), signal)
    # nearest neighbor interpolation
    signal = signal[(np.linspace(0, len(signal) - 1, targetLength)).astype(int)]
    
    received = signal[:FILTER_BLOCKS * BLOCK_LENGTH]
    signal = signal[FILTER_BLOCKS * BLOCK_LENGTH:]
    
    signal_power = np.mean(signal ** 2)
    snr = signal_power / noise_power
    print(f"Signal Power: {signal_power}, Noise Power: {noise_power}, SNR: {snr}")

    sent = get_filter_blocks()

    if noise_power == 0: return signal
    return remove_channel(signal, sent, received, snr)

def remove_channel(signal: np.ndarray, sent: np.ndarray, received: np.ndarray, snr: float) -> np.ndarray:
    """
    Remove the channel effect from the received signal using the sent signal.
    """
    filterLength = BLOCK_LENGTH * FILTER_BLOCKS // FILTER_DIVISOR
    filter = np.zeros(filterLength, dtype=complex)
    for i in range(0, BLOCK_LENGTH * FILTER_BLOCKS, filterLength):
        filter += get_filter(sent[i:i + filterLength], received[i:i + filterLength], 1) / FILTER_DIVISOR
    h = np.fft.ifft(filter).real
    h = h[:filterLength // 2]
    plt.plot(h)
    plt.show()
    return np.convolve(signal, h, mode='same')

if __name__ == "__main__":
    AUDIO_PATH = "Downing College.m4a"
    signal = load_audio_file(AUDIO_PATH)
    signal = synchronize(signal)

    sent = get_symbols_from_bitstream(DATA)
    received = decode(signal)

    plot_sent_received_constellation(sent, received)

    # received_data = get_bitstream_from_symbols(received_symbols)[:len(DATA)]
    # print(f'Error Rate: {np.sum(np.array(list(received_data)) != np.array(list(DATA))) / len(DATA) * 100:.2f}%')
