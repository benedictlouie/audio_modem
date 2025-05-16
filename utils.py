import numpy as np
from scipy.io.wavfile import write
import librosa

symbolsPerBlock = 1024
cyclicPrefix = 32
sampleRate = 44100
symbolRate = 10
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