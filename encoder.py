import numpy as np

from utils import *

def encode_block(symbols: np.ndarray) -> np.ndarray:
    """
    Encode a block of symbols into a time-domain signal using IFFT and add a cyclic prefix.
    """
    symbols = np.concatenate((np.zeros(1), symbols, np.zeros(1), np.conjugate(symbols[::-1])))
    symbolsInTime = np.fft.ifft(symbols).real
    return np.concatenate((symbolsInTime[-CYCLIC_PREFIX:], symbolsInTime))

def encode(symbols: np.ndarray) -> np.ndarray:
    """
    Encode a bitstream into a time-domain signal using IFFT and add a cyclic prefix.
    """
    symbols = np.concatenate((symbols, np.repeat(1 + 1j, (-len(symbols)) % SYMBOLS_PER_BLOCK)))
    blockCount = len(symbols) // SYMBOLS_PER_BLOCK
    signal = np.zeros(blockCount * BLOCK_LENGTH)
    for i in range(blockCount):
        signal[i * BLOCK_LENGTH : (i+1) * BLOCK_LENGTH] = encode_block(symbols[i * SYMBOLS_PER_BLOCK : (i+1) * SYMBOLS_PER_BLOCK])
    return signal #/ np.max(np.abs(signal))

def insert_sync_chirp(signal: np.ndarray) -> np.ndarray:
    """
    Insert a synchronization chirp before each block of the signal.
    """
    sync_chirp = get_sync_chirp()
    blocks = signal.reshape(-1, BLOCK_LENGTH)
    blocks_with_chirp = np.zeros((blocks.shape[0], 2*BLOCK_LENGTH))
    blocks_with_chirp[:, :BLOCK_LENGTH] = sync_chirp
    blocks_with_chirp[:, BLOCK_LENGTH:] = blocks
    return blocks_with_chirp.flatten()

def insert_aa_preamble(signal: np.ndarray) -> np.ndarray:
    preamble = get_aa_preamble()
    blocks = signal.reshape(-1, BLOCK_LENGTH)
    blocks_with_preamble = np.zeros((blocks.shape[0], 2*BLOCK_LENGTH))
    blocks_with_preamble[:, :BLOCK_LENGTH] = preamble
    blocks_with_preamble[:, BLOCK_LENGTH:] = blocks
    return blocks_with_preamble.flatten()

def insert_pilot_signals(signal: np.ndarray) -> np.ndarray:
    """
    Insert pilot signals at the beginning and end of the signal.
    """
    pilot = get_pilot_signal()
    return np.concatenate((pilot, signal, pilot[::-1]))

def get_aa_preamble() -> np.ndarray:
    fourier = np.random.RandomState(39).rand(N_DFT//2)
    fourier -= 0.5
    fourier *= 8
    fourier[1::2] = 0
    fourier = np.concatenate((fourier, np.conjugate(fourier)))
    aa = np.fft.ifft(fourier).real
    aa = np.concatenate((aa[-CYCLIC_PREFIX:], aa))
    plt.plot(aa)
    plt.show()
    return aa
    
def get_sync_chirp() -> np.ndarray:
    """
    Generate a synchronization chirp signal with cyclic prefix
    """
    syncLength = 2 * (SYMBOLS_PER_BLOCK + 1)
    t = np.linspace(0, syncLength / SAMPLE_RATE, syncLength)
    chirp = CHIRP_FACTOR * np.sin(2 * np.pi * (CHIRP_LOW + (CHIRP_HIGH - CHIRP_LOW) * t * SAMPLE_RATE / syncLength) * t)
    return np.concatenate((chirp[-CYCLIC_PREFIX:], chirp))

def get_pilot_signal() -> np.ndarray:
    """
    Generate a pilot signal using a chirp signal.
    """
    t = np.linspace(0, CHIRP_TIME, CHIRP_LENGTH)
    chirp = CHIRP_FACTOR * np.sin(2 * np.pi * (CHIRP_LOW + (CHIRP_HIGH - CHIRP_LOW) * t / CHIRP_TIME) * t)
    return chirp
    
if __name__ == "__main__":
    symbols = get_symbols_from_bitstream(DATA)
    signal = encode(symbols)
    signal = insert_sync_chirp(signal)
    # signal = insert_aa_preamble(signal)

    signal = insert_pilot_signals(signal)
    signal = np.concatenate((np.zeros(SAMPLE_RATE), signal, np.zeros(SAMPLE_RATE)))
    write_wav(AUDIO_PATH, signal)

    print(f'Bitrate: {round(len(DATA) * SAMPLE_RATE / (len(signal) - 2 * SAMPLE_RATE))} bps')
    print(f'Bitstream: {len(DATA)} bits')
    print(f'Time: {(len(signal) / SAMPLE_RATE - 2):.2f} seconds')