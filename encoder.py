import numpy as np

from utils import *

def encode_block(symbols):
    zeros = np.zeros(blockLength - cyclicPrefix - 2 * symbolsPerBlock - 2)
    symbols = np.concatenate((np.zeros(1), symbols, np.zeros(1), zeros, np.conjugate(symbols[::-1])))
    symbolsInTime = np.fft.ifft(symbols).real
    return np.concatenate((symbolsInTime[-cyclicPrefix:], symbolsInTime))

def encode(data):
    constellation = {
        '00': 1 + 1j,
        '01': -1 + 1j,
        '10': 1 - 1j,
        '11': -1 - 1j
    }
    symbols = np.array([constellation[data[i:i+2]] for i in range(0, len(data), 2)])
    remainder = len(symbols) % (symbolsPerBlock * syncBlockPeriod)
    if remainder:
        symbols = np.concatenate((symbols, np.repeat(constellation['11'], symbolsPerBlock * syncBlockPeriod - remainder)))

    blockCount = len(symbols) // symbolsPerBlock
    signal = np.zeros(blockCount * blockLength)
    for i in range(blockCount):
        signal[i * blockLength:(i + 1) * blockLength] = encode_block(symbols[i * symbolsPerBlock:(i + 1) * symbolsPerBlock])
    return signal

def insert_sync_blocks(signal):
    syncBlockCount = len(signal) // blockLength // syncBlockPeriod
    syncBlocks = synchronize_blocks(syncBlockCount + 2)
    output = np.zeros(sampleRate)
    for i in range(syncBlockCount):
        output = np.concatenate((output, syncBlocks[i+1], signal[i * syncBlockLength : (i + 1) * syncBlockLength]))
    output = np.concatenate((output, syncBlocks[0], syncBlocks[syncBlockCount + 1], np.zeros(sampleRate)))
    return output

def synchronize_blocks(blockCount):
    sequence = encode(get_non_repeating_bits(2 * blockCount * symbolsPerBlock))
    return [sequence[i * blockLength:(i + 1) * blockLength] for i in range(blockCount)]
    

if __name__ == "__main__":
    data = text_to_binary("Hello World")
    signal = encode(data)
    signal = insert_sync_blocks(signal)
    write_wav(audio_path, signal)