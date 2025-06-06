import numpy as np
from typing import Tuple

from utils.parameters import *
from utils.utils import *
from utils.plot import *

def decode(signal: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Decode the received signal into symbols.
    """
    fourier = np.fft.fft(signal, axis=1) * filter

    # Apply low and high-pass filter to remove problematic frequencies
    symbols = fourier[:, 1+HIGH_PASS_INDEX: 1+LOW_PASS_INDEX].flatten()
    return symbols

def synchronize(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronize the received signal and return the filter.
    Returns 3 matrices with N_DFT columns and an array of length N_DFT.
    """
    chirp_signal = get_chirp()

    # Find start and end indices of the signal by correlation with the chirp
    startIndex = np.argmax(np.correlate(signal, chirp_signal[::-1]))
    endIndex = np.argmax(np.correlate(signal, chirp_signal))
    
    # Estimate number of frames
    num_frames = round((endIndex - startIndex - CHIRP_LENGTH) / ((INFORMATION_BLOCKS_PER_FRAME + 1) * BLOCK_LENGTH))
    print("There are", num_frames, "frames.")
    known_blocks = get_known_blocks(num_frames)
    
    # left_bound is the end of the chirp minus one block length
    left_bound = startIndex + CHIRP_LENGTH - BLOCK_LENGTH

    # sync for each data block using correlation of each known block
    sync_indices = np.array([])
    current_index = 0
    while left_bound + FRAME_LENGTH < endIndex:
        correlation = np.correlate(signal[left_bound:left_bound + FRAME_LENGTH], known_blocks[current_index][CYCLIC_PREFIX:])
        index = int(np.argmax(correlation)) - CYCLIC_PREFIX + left_bound
        sync_indices = np.append(sync_indices, index)
        left_bound = index + FRAME_LENGTH - BLOCK_LENGTH
        current_index += 1

    # indices are the theoretical start indices of each known block - found by adding the frame length to the start index of the first known block
    x = np.concatenate(([startIndex], np.arange(current_index) * FRAME_LENGTH + startIndex + CHIRP_LENGTH, [startIndex + CHIRP_LENGTH + current_index * FRAME_LENGTH]))
    y = np.concatenate(([startIndex], sync_indices, [endIndex]))

    # Find the drift gradient by regression of the theoretical indices and the indices found by synchronization.
    # Using m = ∑xy/∑x² with normalised values, we estiate the slope
    drift_gradient = np.dot(x - np.mean(x), y - np.mean(y)) / np.sum((x - np.mean(x)) ** 2)

    # Using c = (∑y - m∑x) / N, we estimate the intercept
    drift_constant = np.mean(y) - drift_gradient * np.mean(x)

    # frame_indices is a matrix with frameLength columns, every row contains the corrected sample indices
    frame_indices = (np.arange(startIndex + CHIRP_LENGTH, startIndex + CHIRP_LENGTH + current_index * FRAME_LENGTH) * drift_gradient + drift_constant).reshape(-1, FRAME_LENGTH)

    # extarcted_indices is a matrix of the same shape, but rounding every value to an integer
    extracted_indices = np.vstack([np.arange(round(row[0]), round(row[0]) + FRAME_LENGTH) for row in frame_indices]) - SHIFT_BACK
    drift = extracted_indices - frame_indices

    # Extract relevant blocks we want
    received_frames = signal[extracted_indices]

    # Get the known blocks and informations blocks and their drift. All have N_DFT columns right now.
    sent_known_blocks = known_blocks[:, CYCLIC_PREFIX:BLOCK_LENGTH]
    known_block_drift = drift[:, CYCLIC_PREFIX:BLOCK_LENGTH]
    received_known_blocks = received_frames[:, CYCLIC_PREFIX:BLOCK_LENGTH]

    received_information_frame = received_frames[:, BLOCK_LENGTH:] # BLOCK_LENGTH * INFORMATION_BLOCKS_PER_FRAME columns
    received_information_blocks = np.reshape(received_information_frame, (-1, BLOCK_LENGTH)) # BLOCK_LENGTH columns

    # Remove cyclic prefix
    received_information_blocks = received_information_blocks[:, CYCLIC_PREFIX:] # N_DFT columns

    information_block_drift = drift[:, BLOCK_LENGTH:]
    information_block_drift = np.reshape(information_block_drift, (-1, BLOCK_LENGTH))[:, CYCLIC_PREFIX:] # N_DFT columns

    channel_coefficients, noise_var, snr = estimate_channel_coefficients(sent_known_blocks, received_known_blocks, known_block_drift)
    filter = estimate_filter(channel_coefficients, information_block_drift, snr)
    return received_information_blocks, channel_coefficients, filter, noise_var

def estimate_channel_coefficients(sent_known_blocks: np.ndarray,
                                    received_known_blocks: np.ndarray,
                                    known_block_drift: np.ndarray
                                    ) -> np.ndarray:
    """
    Estimate channel coefficients from a known sent and received block.
    Returns a matrix with N_DFT columns and 2 arrays with length N_DFT
    """

    # FFT of sent and received known blocks across every row, each row has N_DFT columns
    sent_fourier = np.fft.fft(sent_known_blocks, axis=1)
    received_fourier = np.fft.fft(received_known_blocks, axis=1)

    # Unrotate the constellation clockwise by 2πkt/N
    # received_fourier has N_DFT columns
    received_fourier *= np.exp(-2j * np.pi * np.arange(N_DFT) * known_block_drift / N_DFT)

    # Estimate the channel coefficient with no rotation
    # channel_coefficient_estimate has N_DFT columns
    channel_coefficient_estimate = received_fourier / (sent_fourier + 1e-10)

    # noise is equal to Y - HX
    mean_channel_coefficient = np.mean(channel_coefficient_estimate, axis=0)
    noise = mean_channel_coefficient * sent_fourier - received_fourier
    noise_var = np.mean(np.abs(noise) ** 2, axis=0)

    # Estimate SNR from signal_power / noise_var
    signal_power = np.mean(np.abs(sent_fourier) ** 2, axis=0)
    snr = signal_power / noise_var
    
    # Compute noise variance for LDPC coding
    noise_var = np.mean(np.real(noise) ** 2 + np.imag(noise) ** 2, axis=0) / 2

    return channel_coefficient_estimate, noise_var, snr

def estimate_filter(channel_coefficient_estimate: np.ndarray, information_block_drift: np.ndarray, snr):
    """
    Estimate filter from channel_coefficients with MMSE (Wiener) filter formula
    Returns a filter matrix with N_DFT columns and repeated rows.
    """

    # Moving average
    # radius = 1
    # for i in range(len(channel_coefficient_estimate)):
    #     channel_coefficient_estimate[i, :] = np.mean(channel_coefficient_estimate[max(0, i-radius): i+radius+1, :], axis=0)

    # MMSE (Wiener) filter formula
    filter = np.conjugate(channel_coefficient_estimate) / (np.abs(channel_coefficient_estimate) ** 2 + 1 / snr)

    # The final filter is the mean of all filters, then include the unrotation in the filter
    filter = np.mean(filter, axis=0) * np.exp(-2j * np.pi * np.arange(N_DFT) * information_block_drift / N_DFT)
    return filter

def estimate_ldpc_noise_variance(channel_coefficients: np.ndarray, sigma2):
    """
    FOR LDPC DECODING
    Returns sigma_k^2, an array of length SYMBOLS_PER_BLOCK.
    """
    # Find the mean of channel_coefficients across DFT blocks
    # After that, channel_coefficients has length N_DFT
    channel_coefficients = np.mean(channel_coefficients, axis=0)
    magnitude_squared = np.abs(channel_coefficients) ** 2
    sigmak2 = sigma2 / magnitude_squared
    return sigmak2[1+HIGH_PASS_INDEX: 1+LOW_PASS_INDEX]

if __name__ == "__main__":
    signal = load_audio_file(RECEIVED_AUDIO_PATH)

    received_information_blocks, channel_coefficients, filter, noise_var = synchronize(signal)
    received_symbols = decode(received_information_blocks, filter)

    ldpc_noise_variance = estimate_ldpc_noise_variance(channel_coefficients, noise_var)
    received_data = get_bitstream_from_symbols(received_symbols, ldpc_noise_variance)

    if KNOWN_RECEIVER:
        original_bits = get_original_bits()
        sent_symbols = get_symbols_from_bitstream(original_bits)
        plot_sent_received_constellation(sent_symbols, received_symbols)
        plot_error_per_bin(received_symbols, sent_symbols, filter)
        received_data = received_data[:len(original_bits)]
        print(f'Bit Error Rate after LDPC: {np.sum(received_data != original_bits) / len(original_bits) * 100:.2f}%')
    else:
        plot_received_constellation(received_symbols)

    path = decode_bits_to_file(received_data)