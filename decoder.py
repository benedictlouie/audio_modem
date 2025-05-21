import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from utils import *
from encoder import get_pilot_signal, get_sync_chirp, get_aa_preamble

def decode_block(data: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Decode a block of data using FFT and extract the symbols.
    """
    fourier = np.fft.fft(data[CYCLIC_PREFIX:])
    fourier *= filter
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
    print(startIndex, endIndex)
    
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

def synchronizeAA(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pilot = get_pilot_signal()
    startIndex = np.argmax(np.correlate(signal, pilot))
    endIndex = np.argmax(np.correlate(signal, pilot[::-1]))
    
    plt.plot(np.convolve(signal, np.ones(300)/300, mode='same'))
    plt.show()

    output = np.array([])
    leftBound = startIndex + CHIRP_LENGTH
    print(startIndex, CHIRP_LENGTH)

    while leftBound < endIndex:
        cor = []
        for i in range(leftBound - BLOCK_LENGTH, leftBound + BLOCK_LENGTH):
            P = np.sum(signal[i: i+N_DFT//2] * signal[i+N_DFT//2: i+N_DFT])
            R = np.sum(signal[i+N_DFT//2: i+N_DFT] ** 2)
            cor.append(abs(P)**2 / R**2)
        plt.plot(cor)
        kernel = np.ones(CYCLIC_PREFIX) / CYCLIC_PREFIX
        cor = np.convolve(cor, kernel, mode='same')
        plt.plot(cor)
        plt.plot(signal[leftBound-BLOCK_LENGTH: leftBound+BLOCK_LENGTH])
        plt.show()

        # leftBound = leftBound - BLOCK_LENGTH + np.argmax(cor) - CYCLIC_PREFIX//2 + BLOCK_LENGTH
        leftBound += BLOCK_LENGTH
        # print(np.argmax(cor), BLOCK_LENGTH, leftBound)
        output = np.concatenate((output, signal[leftBound: leftBound+BLOCK_LENGTH]))
        leftBound += BLOCK_LENGTH

    # noise_power = np.mean(signal[:startIndex] ** 2)
    # signal_power = np.mean(output ** 2)
    # snr = signal_power / noise_power
    # snr_db = 10 * np.log10(snr)
    # print(f'noise power: {noise_power}, signal power: {signal_power}, SNR: {snr_db:.2f} dB')
    # received = np.reshape(received, (-1, BLOCK_LENGTH))

    return output, np.ones(N_DFT)


def estimate_filter(sent: np.ndarray, received: np.ndarray, snr: float) -> np.ndarray:
    """
    Remove the channel effect from the received signal using the sent signal.
    """
    S = np.fft.fft(sent)
    R = np.fft.fft(received)
    H = R / (S + 1e-10)
    # plt.plot((1/H[0]).real)
    # plt.plot((1/H[0]).imag)
    # plt.show()
    filter = np.conjugate(H) / (np.abs(H) ** 2 + 1 / snr)
    filter = np.mean(filter, axis=0)
    plt.plot(filter.real)
    plt.plot(filter.imag)
    plt.show()
    return filter

if __name__ == "__main__":
    RECEIVED_AUDIO = AUDIO_PATH
    # RECEIVED_AUDIO = "noisy.wav"
    # RECEIVED_AUDIO = "Downing College.m4a"
    
    signal = load_audio_file(RECEIVED_AUDIO)
    signal, filter = synchronize(signal)
    # signal, filter = synchronizeAA(signal)

    sent_symbols = get_symbols_from_bitstream(DATA)
    received_symbols = decode(signal, filter)[:len(sent_symbols)]

    if len(received_symbols) != len(sent_symbols): exit('synchronization failed')

    plot_sent_received_constellation(sent_symbols, received_symbols)

    print("Constellation correctness:", np.sum([decode_constellation(rs) == decode_constellation(ss) for rs, ss in zip(received_symbols, sent_symbols)]) / len(sent_symbols))

    received_data = get_bitstream_from_symbols(received_symbols)[:len(DATA)]
    print("Bit error rate:", np.sum([rd != sd for rd,sd in zip(received_data, DATA)])/ len(DATA))