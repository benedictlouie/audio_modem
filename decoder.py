import numpy as np
from typing import Tuple

from encoder import encode
from utils.parameters import *
from utils.utils import *
from utils.plot import *

def iterative_decoder(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y, x, known_blocks, startIndex, num_frames = synchronize(signal)
    max_iter = [True] * num_frames * INFORMATION_BLOCKS_PER_FRAME * 2

    while True:
        decoded_blocks_bool = [not (max_iter[i] or max_iter[i + 1]) for i in range(0, len(max_iter), 2)]

        # Find indices of decoded information blocks
        new_x = x.copy()
        new_y = y.copy()
        for frame_count in range(num_frames):
            for info_count in range(INFORMATION_BLOCKS_PER_FRAME):
                count = frame_count * INFORMATION_BLOCKS_PER_FRAME + info_count
                if not decoded_blocks_bool[count]: continue
                sync_index = startIndex + CHIRP_LENGTH + frame_count * FRAME_LENGTH + (info_count + 1) * BLOCK_LENGTH
                new_x = np.append(new_x, sync_index)
                sync_signal = encoded_blocks[count]
                left_bound = sync_index - BLOCK_LENGTH // 2
                right_bound = sync_index + 3 * BLOCK_LENGTH // 2
                
                found_index = np.argmax(np.correlate(signal[left_bound:right_bound], sync_signal)) + left_bound
                new_y = np.append(new_y, found_index)

        outlier_count = 0
        while True:
            if len(new_x) < 2: exit('bad synchronization')
            drift_gradient = np.dot(new_x - np.mean(new_x), new_y - np.mean(new_y)) / np.sum((new_x - np.mean(new_x)) ** 2)
            drift_constant = np.mean(new_y) - drift_gradient * np.mean(new_x)

            pred_y = drift_gradient * new_x + drift_constant
            residuals = np.abs(new_y - pred_y)
            mse = np.mean(residuals ** 2)
            
            if mse < 1:
                break

            # remove the largest residual
            max_index = np.argmax(residuals)
            new_x = np.delete(new_x, max_index)
            new_y = np.delete(new_y, max_index)
            outlier_count += 1
        # print(f"Outlier count: {outlier_count}")


        frame_indices = (np.arange(startIndex + CHIRP_LENGTH, startIndex + CHIRP_LENGTH + num_frames * FRAME_LENGTH) * drift_gradient + drift_constant).reshape(-1, FRAME_LENGTH)

        extracted_indices = np.vstack([np.arange(round(row[0]), round(row[0]) + FRAME_LENGTH) for row in frame_indices]) - SHIFT_BACK
        drift = extracted_indices - frame_indices

        received_frames = signal[extracted_indices]
        
        sent_known_blocks = known_blocks[:, :BLOCK_LENGTH]
        received_known_blocks = received_frames[:, :BLOCK_LENGTH]
        known_block_drift = drift[:, :BLOCK_LENGTH]

        received_information_frame = received_frames[:, BLOCK_LENGTH:]
        received_information_blocks = np.reshape(received_information_frame, (-1, BLOCK_LENGTH))
        information_block_drift = drift[:, BLOCK_LENGTH:].reshape(-1, BLOCK_LENGTH)

        # If any decoded block, add to the channel estimation
        if any(decoded_blocks_bool):
            sent_known_blocks = np.vstack([sent_known_blocks, encoded_blocks[decoded_blocks_bool]])
            received_known_blocks = np.vstack([received_known_blocks, received_information_blocks[decoded_blocks_bool]])
            known_block_drift = np.vstack([known_block_drift, information_block_drift[decoded_blocks_bool]])

        channel_coefficients, snr = estimate_channel_coefficients(sent_known_blocks, received_known_blocks, known_block_drift)
        filter = estimate_filter(channel_coefficients, information_block_drift, snr)

        new_received_symbols = decode(received_information_blocks, filter)
        ldpc_noise_variance = estimate_ldpc_noise_variance(new_received_symbols)
        new_received_data, new_max_iter = get_bitstream_from_symbols(new_received_symbols, ldpc_noise_variance)

        if sum(max_iter) < sum(new_max_iter):
            return received_data, received_symbols
        elif sum(max_iter) == sum(new_max_iter):
            return new_received_data, new_received_symbols
        
        received_symbols = new_received_symbols.copy()
        received_data = new_received_data.copy()
        
        max_iter = new_max_iter.copy()
        encoded_symbols = get_symbols_from_bitstream(received_data)
        encoded_blocks = encode(encoded_symbols).reshape(-1, BLOCK_LENGTH)


def decode(signal: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Decode the received signal into symbols.
    """
    fourier = np.fft.fft(signal[:, CYCLIC_PREFIX:], axis=1) * filter

    # Apply low and high-pass filter to remove problematic frequencies
    symbols = fourier[:, 1+HIGH_PASS_INDEX: 1+LOW_PASS_INDEX].flatten()
    return symbols

def synchronize(signal: np.ndarray):
    """
    Synchronize the received signal and return the filter.
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

    for frame_count in range(num_frames):
        frame_index = startIndex + CHIRP_LENGTH + frame_count * FRAME_LENGTH
        sync_index = frame_index
        x = np.append(x, sync_index)

        sync_signal = known_blocks[frame_count]
        left_bound = sync_index - BLOCK_LENGTH // 2
        right_bound = sync_index + 3 * BLOCK_LENGTH // 2
        
        found_index = np.argmax(np.correlate(signal[left_bound:right_bound], sync_signal)) + left_bound
        y = np.append(y, found_index)

    return y, x, known_blocks, startIndex, num_frames

def estimate_channel_coefficients(sent_known_blocks: np.ndarray,
                                    received_known_blocks: np.ndarray,
                                    known_block_drift: np.ndarray,
                                    ) -> np.ndarray:
    """
    Estimate channel coefficients from a known sent and received block.
    Returns a matrix with N_DFT columns and 2 arrays with length N_DFT
    """
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
    noise = sent_fourier * mean_channel_coefficient - received_fourier
    noise_power = np.mean(np.abs(noise) ** 2, axis=0)

    # Estimate SNR from signal_power / noise_var
    signal_power = np.mean(np.abs(sent_fourier) ** 2, axis=0)
    snr = signal_power / noise_power

    return channel_coefficient_estimate, snr

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

def estimate_ldpc_noise_variance(received_symbols: np.ndarray) -> float:

    received_symbols = np.reshape(received_symbols, (-1, SYMBOLS_PER_BLOCK))
    mean_distance_squared = np.zeros(SYMBOLS_PER_BLOCK)
    targets = [1+1j, 1-1j, -1+1j, -1-1j]
    for k in range(SYMBOLS_PER_BLOCK):
        close_symbols = []
        for target in targets:
            distances = np.abs(received_symbols[:, k] - target)
            close_symbols.extend(received_symbols[:, k][distances < 1] - target)
        if close_symbols: mean_distance_squared[k] = np.mean(np.abs(close_symbols) ** 2)
    mean_distance_squared[mean_distance_squared == 0] = np.mean(mean_distance_squared[mean_distance_squared > 0])
    return mean_distance_squared


if __name__ == "__main__":
    signal = load_audio_file(RECEIVED_AUDIO_PATH)

    received_data, received_symbols = iterative_decoder(signal)

    if KNOWN_RECEIVER:
        original_bits = get_original_bits()
        sent_symbols = get_symbols_from_bitstream(original_bits)
        plot_sent_received_constellation(sent_symbols, received_symbols)
        received_data = received_data[:len(original_bits)]
        print(f'Bit Error Rate after LDPC: {np.sum(received_data != original_bits) / len(original_bits) * 100:.2f}%')
    else:
        plot_received_constellation(received_symbols)

    path = decode_bits_to_file(received_data)
    open_file_with_default_app(path)