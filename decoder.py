import numpy as np
import matplotlib.pyplot as plt

from utils import *
from encoder import get_start_block, get_sync_block

# Decoding 4-QAM constellation with Gray Code
def decode_constellation(z):
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

global channel_coefficients

# Remove cyclic prefix, do FFT then remove zero paddings
def decode_block(data):
    fourier = np.fft.fft(data[cyclicPrefix:])[1:symbolsPerBlock + 1]
    if channel_coefficients != None:
        fourier /= channel_coefficients
    return fourier

def decode(data):
    
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
    repeatCount = sampleRate // symbolRate
    bits = ''.join([decode_constellation(np.mean(symbols[i:i+repeatCount])) for i in range(0, len(symbols), repeatCount)])
    return bits

def synchronize(signal):
    startBlock = get_start_block()
    syncBlock = get_sync_block()

    startCorrelation = np.abs(np.correlate(signal, startBlock))
    startIndex = np.argmax(startCorrelation)

    bitstream = get_non_repeating_bits(2 * symbolsPerBlock)
    symbols = np.array([constellation[bitstream[i:i+2]] for i in range(0, len(bitstream), 2)])
    print(symbols[:5])
    symbols_recieved = decode_block(signal[startIndex: startIndex + blockLength])
    print(symbols_recieved[:5])
    channel_coefficients = symbols_recieved / symbols
    print(channel_coefficients[:5])

    syncCorrelation = np.abs(np.correlate(signal, syncBlock))
    syncIndices = np.array([startIndex])
    syncLength = syncBlockPeriod * blockLength
    print(syncLength)
    leftBound = startIndex + syncLength // 2
    while leftBound + syncLength < len(signal):
        syncIndices = np.append(syncIndices, leftBound + np.argmax(syncCorrelation[leftBound : leftBound + syncLength]))
        leftBound = syncIndices[-1] + syncLength // 2
    print(syncIndices - startIndex)
    
    output = np.array([])
    for i in syncIndices:
        i = int(i)
        output = np.concatenate((output, signal[i + blockLength : i + blockLength + syncLength]))
    return output




if __name__ == "__main__":
    audio_path = "Downing College.m4a"
    signal = load_audio_file(audio_path)
    signal = synchronize(signal)
    decodedData = decode(signal)
    print(binary_to_text(decodedData))
