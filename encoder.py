import numpy as np
from scipy.io.wavfile import write

blockLength = 1024
cyclicPrefix = 32
frequency = 500

def fibonacci_binary_bits(n):
    fib1, fib2 = 0, 1
    binary_string = ''
    
    while len(binary_string) < n:
        binary_string += bin(fib1)[2:]
        fib1, fib2 = fib2, fib1 + fib2

    return binary_string[:n]

def encode_block(data):
    constellation = {
        '00': 1 + 1j,
        '01': -1 + 1j,
        '10': 1 - 1j,
        '11': -1 - 1j
    }
    symbols = np.array([constellation[data[i:i+2]] for i in range(0, len(data), 2)])
    symbols = np.concatenate((np.zeros(1), symbols, np.zeros(1), np.conjugate(symbols[::-1])))
    symbolsInTime = np.fft.ifft(symbols).real
    return np.concatenate((symbolsInTime[-cyclicPrefix:], symbolsInTime))

def write_wav(filename, data, sample_rate=frequency):
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

if __name__ == "__main__":
    data = fibonacci_binary_bits(blockLength - 2)
    print(data)
    amplitude = encode_block(data)
    print(amplitude)
    write_wav("output.wav", amplitude)

