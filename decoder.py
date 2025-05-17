import numpy as np
import matplotlib.pyplot as plt

from utils import *
from encoder import get_sync_blocks

# Decoding 4-QAM constellation with Gray Code
def decode_constellation(z: complex) -> str:
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

channel_coefficients = None

# Remove cyclic prefix, do FFT, then remove zero paddings
def decode_block(data: np.ndarray) -> np.ndarray:
    fourier = np.fft.fft(data[cyclicPrefix:])[1:symbolsPerBlock + 1]

    # Channel coefficients -- not working
    # try: fourier /= channel_coefficients
    # except: print("Estimating channel coefficients...")

    return fourier

def decode(data: np.ndarray) -> str:
    
    # Pad zeros if there is remainder
    remainder = len(data) % blockLength
    if remainder:
        print(f'Warning: remainder = {remainder}')
        data = np.concatenate((data, np.zeros(blockLength - remainder)))
    
    # We have this many DFT blocks
    blockCount = len(data) // blockLength

    # Initialise symbols array
    symbols = np.zeros(blockCount * symbolsPerBlock, dtype=complex)

    # For every block, we decode the signal
    for i in range(blockCount):
        symbols[i * symbolsPerBlock: (i+1) * symbolsPerBlock] = decode_block(data[i * blockLength: (i+1) * blockLength])

    # Repetition decoding, take the mean
    bits = ''.join([decode_constellation(np.mean(symbols[i:i+repeatCount])) for i in range(0, len(symbols), repeatCount)])
    return bits

def remove_channel(signal:   np.ndarray,
                   sent:     np.ndarray,
                   received: np.ndarray,
                   snr_db:   float = snr_db,
                   eps: float = 1e-12) -> np.ndarray:
    # FFT length — power of two ≥ longer of signal or pilot for speed/linearity
    n_fft = 1 << (max(len(signal), len(sent)) - 1).bit_length()

    # Least-squares channel estimate Ĥ(f) = R(f) / S(f)
    S = np.fft.rfft(sent,     n=n_fft)
    R = np.fft.rfft(received, n=n_fft)
    H_ls = R / (S + eps)

    # MMSE equaliser G(f) = H*(f) / (|H(f)|² + N₀/Pₓ)
    snr_lin  = 10 ** (snr_db / 10)
    n0_over_px = 1.0 / snr_lin                  # N₀ / Pₓ

    G = np.conj(H_ls) / (np.abs(H_ls)**2 + n0_over_px + eps)

    # Equalise the long recording
    Sig = np.fft.rfft(signal, n=n_fft)
    equalised = np.fft.irfft(Sig * G, n=n_fft)

    return equalised[:len(signal)]

def synchronize(signal: np.ndarray) -> np.ndarray:

    startBlock, syncBlock, endBlock = get_sync_blocks()

    # Find where the signal starts and ends
    startCorrelation, endCorrelation = np.correlate(signal, startBlock), np.correlate(signal, endBlock)
    startIndex, endIndex = np.argmax(startCorrelation), np.argmax(endCorrelation)

    # Remove channel from the signal
    signal = remove_channel(signal, startBlock, signal[startIndex : startIndex + blockLength * startEndBlockMultiplier])

    # Find each sync block
    syncCorrelation = np.correlate(signal, syncBlock)
    
    syncIndices = np.array([startIndex + (startEndBlockMultiplier - 1) * blockLength])
    leftBound = startIndex + syncLength // 2
    while leftBound + syncLength < len(signal):
        found_index = leftBound + np.argmax(syncCorrelation[leftBound : leftBound + syncLength])
        if found_index > endIndex:
            break
        syncIndices = np.append(syncIndices, found_index)
        leftBound = syncIndices[-1] + syncLength // 2
    syncIndices = syncIndices[:-1]  # Remove the last one, which is out of bounds
    # Remove all sync blocks
    output = np.array([])
    for i in syncIndices:
        i = int(i)
        output = np.concatenate((output, signal[i + blockLength : i + blockLength + syncLength]))

    return output

if __name__ == "__main__":
    # audio_path = "Downing College.m4a"
    signal = load_audio_file(audio_path)
    signal = synchronize(signal)
    data = decode(signal)
    # data = decode_ldpc(data)
    print(binary_to_text(data))
