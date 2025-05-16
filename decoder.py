import numpy as np
import matplotlib.pyplot as plt

from utils import *
from encoder import synchronize_blocks

def decode_constellation(z):
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

def decode_block(data):
    return np.fft.fft(data[cyclicPrefix:])[1:symbolsPerBlock + 1]

def decode(data):
    blockCount = len(data) // blockLength
    symbols = np.zeros(blockCount * symbolsPerBlock, dtype=complex)
    for i in range(blockCount):
        symbols[i * symbolsPerBlock:(i + 1) * symbolsPerBlock] = decode_block(data[i * blockLength:(i + 1) * blockLength])
    bitstream = ''.join([decode_constellation(symbol) for symbol in symbols])
    return bitstream

def get_channel(signal):
    pass

def synchronize(signal):
    syncBlockCount = len(signal) // blockLength // syncBlockPeriod
    syncBlocks = synchronize_blocks(syncBlockCount)

    end_filter = syncBlocks[0]
    end_filter_match = np.correlate(signal, end_filter)
    end_filter_index = np.argmax(end_filter_match)

    output = np.array([])
    sync_index = 1
    while True:
        filter = syncBlocks[sync_index]
        filter_match = np.correlate(signal, filter)
        # plt.plot(filter_match)
        # plt.title(f'Filter Match {sync_index}')
        # plt.show()
        match_index = np.argmax(np.abs(filter_match))

        if match_index > end_filter_index:
            break

        output = np.concatenate((output, signal[match_index + blockLength: match_index + blockLength + syncBlockLength]))
        sync_index += 1
    return output


if __name__ == "__main__":
    audio_path = "Downing College.m4a"
    signal = load_audio_file(audio_path)
    signal = synchronize(signal)
    decodedData = decode(signal)
    print(binary_to_text(decodedData))