import numpy as np

from utils import *

def decode_constellation(z):
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

def decode_block(data):
    return np.fft.fft(data[cyclicPrefix:])[1:symbolsPerBlock + 1]

def decode(data):
    blockLength = 2 * (symbolsPerBlock + 1) + cyclicPrefix
    blockCount = len(data) // blockLength
    symbols = np.zeros(blockCount * symbolsPerBlock, dtype=complex)
    for i in range(blockCount):
        symbols[i * symbolsPerBlock:(i + 1) * symbolsPerBlock] = decode_block(data[i * blockLength:(i + 1) * blockLength])
    repeatCount = sampleRate // symbolRate
    bits = ''.join([decode_constellation(np.sum(symbols[i:i+repeatCount])) for i in range(0, len(symbols), repeatCount)])
    return bits.split(prefix)[1]

if __name__ == "__main__":
    y = load_audio_file(audio_path)
    decodedData = decode(y)
    print(binary_to_text(decodedData))