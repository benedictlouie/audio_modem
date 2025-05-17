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

channel_coefficients = None

# Remove cyclic prefix, do FFT, then remove zero paddings
def decode_block(data):
    fourier = np.fft.fft(data[cyclicPrefix:])[1:symbolsPerBlock + 1]

    # Channel coefficients -- not working
    # try: fourier /= channel_coefficients
    # except: print("Estimating channel coefficients...")

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

    # Find where the signal starts
    startCorrelation = np.correlate(signal, startBlock)
    startIndex = np.argmax(startCorrelation)
    # plt.plot(startCorrelation)
    # plt.show()

    # Estimate channel coefficients
    bitstream = get_non_repeating_bits(2 * symbolsPerBlock)
    symbols = np.array([constellation[bitstream[i:i+2]] for i in range(0, len(bitstream), 2)])
    symbols_recieved = decode_block(signal[startIndex: startIndex + blockLength])
    global channel_coefficients
    channel_coefficients = symbols_recieved / symbols
    # print(channel_coefficients[:5])

    # Find each sync block
    syncCorrelation = np.correlate(signal, syncBlock)
    
    syncIndices = np.array([startIndex])
    syncLength = syncBlockPeriod * blockLength
    leftBound = startIndex + syncLength // 2
    while leftBound + syncLength < len(signal):
        syncIndices = np.append(syncIndices, leftBound + np.argmax(syncCorrelation[leftBound : leftBound + syncLength]))
        leftBound = syncIndices[-1] + syncLength // 2
    # print(len(syncIndices), syncIndices - startIndex)
    # plt.plot(syncCorrelation)
    # plt.show()
    
    # Remove all sync blocks
    output = np.array([])
    for i in syncIndices:
        i = int(i)
        output = np.concatenate((output, signal[i + blockLength : i + blockLength + syncLength]))
    return output

if __name__ == "__main__":
    audio_path = "x.m4a"
    signal = load_audio_file(audio_path)
    signal = synchronize(signal)
    decodedData = decode(signal)
    print(binary_to_text(decodedData))
