import numpy as np
from typing import Tuple

from encoder import encode
from utils.parameters import *
from utils.utils import *
from utils.plot import *

def iterative_decoder(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    received_frames, drift, known_blocks = synchronize(signal)
    
    sent_known_blocks = known_blocks[:, :BLOCK_LENGTH]
    received_known_blocks = received_frames[:, :BLOCK_LENGTH]
    known_block_drift = drift[:, :BLOCK_LENGTH]

    received_information_frame = received_frames[:, BLOCK_LENGTH:] # BLOCK_LENGTH * INFORMATION_BLOCKS_PER_FRAME columns
    received_information_blocks = np.reshape(received_information_frame, (-1, BLOCK_LENGTH)) # BLOCK_LENGTH columns
    information_block_drift = drift[:, BLOCK_LENGTH:].reshape(-1, BLOCK_LENGTH)

    channel_coefficients, noise_var, snr = estimate_channel_coefficients(sent_known_blocks, received_known_blocks, known_block_drift)
    filter = estimate_filter(channel_coefficients, information_block_drift, snr)

    received_symbols = decode(received_information_blocks, filter)
    ldpc_noise_variance = estimate_ldpc_noise_variance(channel_coefficients, noise_var)
    received_data, max_iter = get_bitstream_from_symbols(received_symbols, ldpc_noise_variance)

    # return received_data, received_symbols

    while True:
        old_max_iter_count = sum(max_iter)
        decoded_blocks_bool = [not (max_iter[i] or max_iter[i + 1]) for i in range(0, len(max_iter), 2)]

        encoded_symbols = get_symbols_from_bitstream(received_data)
        encoded_blocks = encode(encoded_symbols).reshape(-1, BLOCK_LENGTH)
        encoded_channel_coefficients = estimate_channel_coefficients(encoded_blocks[decoded_blocks_bool],
                                                    received_information_blocks[decoded_blocks_bool],
                                                    information_block_drift[decoded_blocks_bool])[0]

        new_channel_coefficients = np.vstack((channel_coefficients, encoded_channel_coefficients))
        new_filter = estimate_filter(new_channel_coefficients, information_block_drift, snr)
        received_symbols = decode(received_information_blocks, new_filter)
        received_data, max_iter = get_bitstream_from_symbols(received_symbols, ldpc_noise_variance)
        
        new_max_iter_count = sum(max_iter)
        if new_max_iter_count == old_max_iter_count:
            break

    return received_data, received_symbols


def decode(signal: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Decode the received signal into symbols.
    """
    fourier = np.fft.fft(signal[:, CYCLIC_PREFIX:], axis=1) * filter

    # Apply low and high-pass filter to remove problematic frequencies
    symbols = fourier[:, 1+HIGH_PASS_INDEX: 1+LOW_PASS_INDEX].flatten()
    return symbols

def synchronize(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    x = np.array([startIndex, startIndex + CHIRP_LENGTH + num_frames * FRAME_LENGTH])
    y = np.array([startIndex, endIndex])

    points = BLOCK_LENGTH // SYNCHRONIZATION_LENGTH
    for frame_count in range(num_frames):
        frame_index = startIndex + CHIRP_LENGTH + frame_count * FRAME_LENGTH
        for sync_count in range(points):
            sync_index = frame_index + sync_count * SYNCHRONIZATION_LENGTH
            x = np.append(x, sync_index)

            sync_signal = known_blocks[frame_count, sync_count * SYNCHRONIZATION_LENGTH: (sync_count + 1) * SYNCHRONIZATION_LENGTH]
            left_bound = sync_index - SYNCHRONIZATION_LENGTH // 2
            right_bound = sync_index + 3 * SYNCHRONIZATION_LENGTH // 2
            
            found_index = np.argmax(np.correlate(signal[left_bound:right_bound], sync_signal)) + left_bound
            y = np.append(y, found_index)

    # Find the drift gradient by regression of the theoretical indices and the indices found by synchronization.
    # Using m = ∑xy/∑x² with normalised values, we estimate the slope
    drift_gradient = np.dot(x - np.mean(x), y - np.mean(y)) / np.sum((x - np.mean(x)) ** 2)

    # Using c = (∑y - m∑x) / N, we estimate the intercept
    drift_constant = np.mean(y) - drift_gradient * np.mean(x)

    # frame_indices is a matrix with frameLength columns, every row contains the corrected sample indices
    frame_indices = (np.arange(startIndex + CHIRP_LENGTH, startIndex + CHIRP_LENGTH + num_frames * FRAME_LENGTH) * drift_gradient + drift_constant).reshape(-1, FRAME_LENGTH)

    # extarcted_indices is a matrix of the same shape, but rounding every value to an integer
    extracted_indices = np.vstack([np.arange(round(row[0]), round(row[0]) + FRAME_LENGTH) for row in frame_indices]) - SHIFT_BACK
    drift = extracted_indices - frame_indices

    # Extract relevant blocks we want
    received_frames = signal[extracted_indices]
    return received_frames, drift, known_blocks

def estimate_channel_coefficients(sent_known_blocks: np.ndarray,
                                    received_known_blocks: np.ndarray,
                                    known_block_drift: np.ndarray,
                                    ) -> np.ndarray:
    """
    Estimate channel coefficients from a known sent and received block.
    Returns a matrix with N_DFT columns and 2 arrays with length N_DFT
    """
    if USE_CYCLIC_PREFIX_FOR_FILTER:
        sent_known_blocks = np.vstack((sent_known_blocks[:, CYCLIC_PREFIX:], sent_known_blocks[:, :-CYCLIC_PREFIX]))
        received_known_blocks = np.vstack((received_known_blocks[:, CYCLIC_PREFIX:], received_known_blocks[:, :-CYCLIC_PREFIX]))
        known_block_drift = np.vstack((known_block_drift[:, CYCLIC_PREFIX:], known_block_drift[:, :-CYCLIC_PREFIX]))
    else:
        sent_known_blocks = sent_known_blocks[:, CYCLIC_PREFIX:]
        received_known_blocks = received_known_blocks[:, CYCLIC_PREFIX:]
        known_block_drift = known_block_drift[:, CYCLIC_PREFIX:]

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

    information_block_drift = information_block_drift[:, CYCLIC_PREFIX:]  # Remove cyclic prefix from drift

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

    received_data, received_symbols = iterative_decoder(signal)

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