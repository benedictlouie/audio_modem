import numpy as np
import matplotlib.pyplot as plt

from utils import *
from encoder import synchronize_blocks

# Decoding 4-QAM constellation with Gray Code
def decode_constellation(z):
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

# Remove cyclic prefix, do FFT then remove zero paddings
def decode_block(data):
    return np.fft.fft(data[cyclicPrefix:])[1:symbolsPerBlock + 1]

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
    syncBlockCount = len(signal) // blockLength // (syncBlockPeriod + 1)
    syncBlocks = synchronize_blocks(syncBlockCount)

    end_filter, start_filter = syncBlocks[0], syncBlocks[1]
    start_filter_match = np.correlate(signal, start_filter)
    start_filter_index = np.argmax(np.abs(start_filter_match))
    
    signal = signal[start_filter_index + blockLength:]
    output = np.array([])
    sync_index = 2

    def resize(arr, new_size):
        if len(arr) < new_size:
            return np.concatenate((arr, np.zeros(new_size - len(arr))))
        else:
            return arr[:new_size]
        
    while True:
        filter = syncBlocks[sync_index]
        filter_match = np.correlate(signal, filter)
        match_index = np.argmax(np.abs(filter_match))

        end_filter_match = np.correlate(signal, end_filter)
        end_match_index = np.argmax(np.abs(end_filter_match))

        if match_index < end_match_index:
            output = np.concatenate((output, resize(signal[:match_index], syncBlockPeriod * blockLength)))
            signal = signal[match_index + blockLength:]
        else:
            signal = signal[:end_match_index]
            output = np.concatenate((output, resize(signal, syncBlockPeriod * blockLength)))
            break
        sync_index += 1
    return output

def plot_correlation(signal, n):
    syncBlockCount = n
    syncBlocks = synchronize_blocks(syncBlockCount)

    for i in range(syncBlockCount):
        filter = syncBlocks[i]
        filter_match = np.correlate(signal, filter)
        plt.plot(filter_match, label=f'Sync Block {i}')
    plt.title('Correlation with Sync Blocks')
    plt.xlabel('Sample Index')
    plt.ylabel('Correlation Value')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # audio_path = "Downing College.m4a"
    signal = load_audio_file(audio_path)
    plot_correlation(signal, 10)
    signal = synchronize(signal)
    decodedData = decode(signal)
    print(binary_to_text(decodedData))