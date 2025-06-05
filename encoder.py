import numpy as np

from utils.utils import *
from utils.parameters import *

def encode(symbols: np.ndarray) -> np.ndarray:
    """
    Encode a bitstream into a time-domain signal using IFFT and add a cyclic prefix in front.
    """

    # Pad until a multiple of SYMBOLS_PER_BLOCK
    constellation = [1+1j, 1-1j, -1-1j, -1+1j]
    paddingSymbols = np.random.default_rng(76).choice(constellation, size=(-len(symbols)) % (SYMBOLS_PER_BLOCK * INFORMATION_BLOCKS_PER_FRAME))
    symbols = np.concatenate((symbols, paddingSymbols))
    symbols = symbols.reshape((-1, SYMBOLS_PER_BLOCK))

    # We need at least 2 frames
    while len(symbols) < INFORMATION_BLOCKS_PER_FRAME * 2:
        paddingSymbols = np.random.default_rng(77).choice(constellation, size=(INFORMATION_BLOCKS_PER_FRAME, SYMBOLS_PER_BLOCK))
        symbols = np.vstack((symbols, paddingSymbols))
    
    # Fill unused frequency bins
    # After that, symbols is a matrix with EFFECTIVE_SYMBOLS_PER_BLOCK columns
    symbols = np.concatenate((np.random.default_rng(78).choice(constellation, size=(symbols.shape[0], HIGH_PASS_INDEX)),
                              symbols,
                              np.random.default_rng(79).choice(constellation, size=(symbols.shape[0], EFFECTIVE_SYMBOLS_PER_BLOCK - LOW_PASS_INDEX)),
                            ), axis=1)
    
    # Add zeros to the zeroth bin and do conjugate symmetry
    # After that, encoded_symbols is a matrix with N_DFT columns
    encoded_symbols = np.concatenate((
        np.zeros((symbols.shape[0], 1)),
        symbols,
        np.zeros((symbols.shape[0], 1)),
        np.conjugate(symbols[:, ::-1])
    ), axis=1)

    # Flatten the signal    
    symbolsInTime = np.fft.ifft(encoded_symbols, axis=1).real
    signal = np.concatenate((symbolsInTime[:, -CYCLIC_PREFIX:], symbolsInTime), axis=1).flatten()
    return signal

def insert_known_blocks(signal: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Insert a synchronization OFDM block before every information block of the signal.
    """
    # With varicable frame length
    known_blocks = get_known_blocks(num_frames)
    blocks = signal.reshape(-1, BLOCK_LENGTH * INFORMATION_BLOCKS_PER_FRAME)
    return np.concatenate((known_blocks, blocks), axis=1).flatten()

def insert_chirps(signal: np.ndarray) -> np.ndarray:
    """
    Insert a reverse chirp at the beginning and chirp at the end of the signal.
    """
    chirp = get_chirp()
    return np.concatenate((chirp[::-1], signal, chirp))
    
if __name__ == "__main__":

    original_bits = get_original_bits()

    symbols = get_symbols_from_bitstream(original_bits)
    signal = encode(symbols)

    # Calculate number of frames
    assert len(signal) % (BLOCK_LENGTH * INFORMATION_BLOCKS_PER_FRAME) == 0
    num_frames = len(signal) // (BLOCK_LENGTH * INFORMATION_BLOCKS_PER_FRAME)
    print("There are", num_frames, "frames.")

    signal = insert_known_blocks(signal, num_frames)
    signal = insert_chirps(signal)

    print(f'Bitrate: {round(len(original_bits) * SAMPLE_RATE / len(signal))} bps')
    print(f'Bitstream: {len(original_bits)} bits')
    print(f'Time: {(len(signal) / SAMPLE_RATE):.2f} seconds')

    # Insert one second of nothing before and after the signal
    signal = np.concatenate((np.zeros(SAMPLE_RATE), signal, np.zeros(SAMPLE_RATE)))
    write_wav(AUDIO_PATH, signal)