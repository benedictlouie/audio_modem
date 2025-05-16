import numpy as np

from utils import *

def encode_block(symbols):
    symbols = np.concatenate((np.zeros(1), symbols, np.zeros(1), np.conjugate(symbols[::-1])))
    symbolsInTime = np.fft.ifft(symbols).real
    return np.concatenate((symbolsInTime[-cyclicPrefix:], symbolsInTime))

def encode(data, repeat=True):
    constellation = {
        '00': 1 + 1j,
        '01': -1 + 1j,
        '10': 1 - 1j,
        '11': -1 - 1j
    }
    symbols = np.array([constellation[data[i:i+2]] for i in range(0, len(data), 2)])
    if repeat:
        symbols = np.repeat(symbols, sampleRate // symbolRate)
    remainder = len(symbols) % symbolsPerBlock
    if remainder:
        symbols = np.concatenate((symbols, np.repeat(constellation['00'], symbolsPerBlock - remainder)))

    blockCount = len(symbols) // symbolsPerBlock
    signal = np.zeros(blockCount * blockLength)
    for i in range(blockCount):
        signal[i * blockLength:(i + 1) * blockLength] = encode_block(symbols[i * symbolsPerBlock:(i + 1) * symbolsPerBlock])
    return signal

def start_end_blocks():
    sequence = encode(fibonacci_binary_bits(4 * symbolsPerBlock), repeat=False)
    start = sequence[:blockLength]
    end = sequence[-blockLength:]
    return start, end

if __name__ == "__main__":
    data = text_to_binary("Hello Gael")
    signal = encode(data)
    start, end = start_end_blocks()
    signal = np.concatenate((np.zeros(sampleRate), start, signal, end, np.zeros(sampleRate)))
    write_wav(audio_path, signal)