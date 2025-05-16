import numpy as np
import matplotlib.pyplot as plt

from utils import *
from encoder import start_end_blocks

def decode_constellation(z):
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

def decode_block(data):
    return np.fft.fft(data[cyclicPrefix:])[1:symbolsPerBlock + 1]

def decode(data):
    remainder = len(data) % blockLength
    if remainder:
        print(f'Warning: remainder = {remainder}')
        data = np.concatenate((data, np.zeros(blockLength - remainder)))
    
    blockCount = len(data) // blockLength
    symbols = np.zeros(blockCount * symbolsPerBlock, dtype=complex)
    for i in range(blockCount):
        symbols[i * symbolsPerBlock:(i + 1) * symbolsPerBlock] = decode_block(data[i * blockLength:(i + 1) * blockLength])
    repeatCount = sampleRate // symbolRate
    bits = ''.join([decode_constellation(np.sum(symbols[i:i+repeatCount])) for i in range(0, len(symbols), repeatCount)])
    return bits

def match_start_end_blocks(data):
    start, end = start_end_blocks()

    start_match = np.correlate(data, start)
    end_match = np.correlate(data, end)

    start_index = np.argmax(np.abs(start_match))
    end_index = np.argmax(np.abs(end_match))
    data = data[start_index + blockLength:end_index]
    return data

if __name__ == "__main__":
    audio_path = "input.m4a"
    y = load_audio_file(audio_path)
    y = match_start_end_blocks(y)
    # y += np.random.normal(0, 0.2, len(y))
    decodedData = decode(y)
    print(binary_to_text(decodedData))