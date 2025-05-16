import numpy as np

from utils import *

def encode_block(symbols):
    symbols = np.concatenate((np.zeros(1), symbols, np.zeros(1), np.conjugate(symbols[::-1])))
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
    symbols = np.repeat(symbols, sampleRate // symbolRate)
    symbols = np.concatenate((symbols, np.repeat(constellation['00'], symbolsPerBlock - len(symbols) % symbolsPerBlock)))
    
    blockCount = len(symbols) // symbolsPerBlock
    blockLength = 2 * (symbolsPerBlock + 1) + cyclicPrefix
    signal = np.zeros(blockCount * blockLength)
    for i in range(blockCount):
        signal[i * blockLength:(i + 1) * blockLength] = encode_block(symbols[i * symbolsPerBlock:(i + 1) * symbolsPerBlock])
    return signal

if __name__ == "__main__":
    data = prefix + text_to_binary("Hello, World!") + prefix
    signal = encode(data)
    write_wav(audio_path, signal)

