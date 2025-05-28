import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from utils import *
from encoder import get_known_blocks, get_chirp

def decode(signal: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Decode the received signal into symbols.
    """
    fourier = np.fft.fft(signal, axis=1) * filter

    # Apply high-pass filter to remove problematic frequencies
    symbols = fourier[:, 1 + HIGH_PASS_INDEX: 1 + HIGH_PASS_INDEX + SYMBOLS_PER_BLOCK].flatten()
    return symbols

def synchronize(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronize the received signal and return the filter.
    """
    chirp_signal = get_chirp()

    # Find start and end indices of the signal by correlation with the chirp
    startIndex = np.argmax(np.correlate(signal, chirp_signal[::-1]))
    endIndex = np.argmax(np.correlate(signal, chirp_signal))
    
    known_blocks = get_known_blocks()

    # TODO: make frame length variable 
    frameLength = 2 * BLOCK_LENGTH
    
    # left_bound is the end of the chirp minus one block length
    left_bound = startIndex + CHIRP_LENGTH - frameLength//2

    # sync for each data block using correlation of each known block
    sync_indices = np.array([])
    current_index = 0
    while left_bound + frameLength < endIndex:
        index = int(np.argmax(np.correlate(signal[left_bound:left_bound + frameLength], known_blocks[current_index]))) + left_bound
        sync_indices = np.append(sync_indices, index)
        left_bound = index + frameLength // 2

        current_index += 1

    # indices are the theoretical start indices of each data block - found by adding the frame length to the start index of the first data block
    indices = np.arange(current_index) * frameLength + startIndex + CHIRP_LENGTH

    # Find the drift gradient by regression of the theoretical indices and the indices found by synchronization.
    # Using m = ∑xy/∑x² with normalised values, we estiate the slope
    drift_gradient = np.dot(indices - np.mean(indices), sync_indices - np.mean(sync_indices)) / np.sum((indices - np.mean(indices)) ** 2)

    # Using c = (∑y - m∑x) / N, we estimate the intercept
    drift_constant = np.mean(sync_indices) - drift_gradient * np.mean(indices)

    # frame_indices is a matrix with frameLength columns, every row contains the corrected sample indices
    frame_indices = (np.arange(startIndex + CHIRP_LENGTH, startIndex + CHIRP_LENGTH + current_index * frameLength) * drift_gradient + drift_constant).reshape(-1, frameLength)

    # extarcted_indices is a matrix of the same shape, but rounding every value to an integer
    extracted_indices = np.vstack([np.arange(round(row[0]), round(row[0]) + frameLength) for row in frame_indices])
    drift = extracted_indices - frame_indices

    # extract relevant blocks we want
    received_blocks = signal[extracted_indices]

    # get the known blocks and informations blocks and their drift
    sent_known_blocks = known_blocks[:, CYCLIC_PREFIX:BLOCK_LENGTH]
    known_block_drift = drift[:, CYCLIC_PREFIX:BLOCK_LENGTH]
    received_known_blocks = received_blocks[:, CYCLIC_PREFIX:BLOCK_LENGTH]
    received_information_blocks = received_blocks[:, BLOCK_LENGTH + CYCLIC_PREFIX:]
    information_block_drift = drift[:, BLOCK_LENGTH + CYCLIC_PREFIX:]

    filter = estimate_filter(sent_known_blocks, received_known_blocks, known_block_drift, information_block_drift, WIENER_SNR)
    return received_information_blocks, filter

def estimate_filter(sent_known_blocks: np.ndarray,
                    received_known_blocks: np.ndarray,
                    known_block_drift: np.ndarray,
                    information_block_drift: np.ndarray, 
                    snr: float
                    ) -> np.ndarray:
    """
    Estimate filter from a known sent and received block.
    """
    bins = BLOCK_LENGTH - CYCLIC_PREFIX

    # FFT of sent and received known blocks
    sent_fourier = np.fft.fft(sent_known_blocks, axis=1)
    received_fourier = np.fft.fft(received_known_blocks, axis=1)

    received_fourier *= np.exp(-2j * np.pi * np.arange(bins) * known_block_drift / bins)
    # variable name 'zero_forcing_filter' is misleading, it should be channel_coefficient_estimate or similar instead
    # Zero forcing filter is the inverse of the channel coefficient estimate
    zero_forcing_filter = received_fourier / (sent_fourier + 1e-10)
    # TODO: snr to be an array since it is different for each bin
    # TODO: this requires a coarse channel coefficient estimate which can then be used in Y = X * H + N to get noise variance for better snr estimate
    filter = np.conjugate(zero_forcing_filter) / (np.abs(zero_forcing_filter) ** 2 + 1 / snr)
    # TODO: need further explanation of this rotation
    filter = np.mean(filter, axis=0) * np.exp(-2j * np.pi * np.arange(bins) * information_block_drift / bins)
    return filter

if __name__ == "__main__":
    AUDIO_PATH = "received.wav"
    signal = load_audio_file(AUDIO_PATH)
    received_information_blocks, filter = synchronize(signal)
    received_symbols = decode(received_information_blocks, filter)

    sent_symbols = get_symbols_from_bitstream(DATA)

    received_symbols = received_symbols[:len(sent_symbols)]
    if len(sent_symbols) != len(received_symbols): exit('synchronization error')
    received_symbols *= np.sqrt(2) / np.mean(np.abs(received_symbols))

    plot_sent_received_constellation(sent_symbols, received_symbols)

    # plot_error_per_bin(received_symbols, sent_symbols, filter)

    received_data = get_bitstream_from_symbols(received_symbols)[:len(DATA)]

    print(f'Bit Error Rate: {np.sum(received_data != DATA) / len(DATA) * 100:.2f}%')