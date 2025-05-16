import numpy as np
from scipy.io.wavfile import write
import librosa

symbolsPerBlock = 511
cyclicPrefix = 32
blockLength = 2 * (symbolsPerBlock + 1) + cyclicPrefix
sampleRate = 48000
symbolRate = 100
audio_path = "output.wav"

def load_audio_file(file_path):
    return librosa.load(file_path, sr=None)[0]

def write_wav(filename, data, sample_rate=sampleRate):
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def fibonacci_binary_bits(n):
    fib1, fib2 = 0, 1
    binary_string = ''
    
    while len(binary_string) < n:
        binary_string += bin(fib1)[2:]
        fib1, fib2 = fib2, fib1 + fib2

    return binary_string[:n]

def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_str):
    return ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))