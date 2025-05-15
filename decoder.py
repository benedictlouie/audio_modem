import numpy as np
import librosa

audio_path = "output.wav"
blockLength = 1024
cyclicPrefix = 32
frequency = 500

def load_audio_file(file_path):
    y, _ = librosa.load(file_path, sr=None)
    return y

def decode_constellation(z):
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

def decode(received):
    truncated = received[cyclicPrefix:]
    filtered = np.fft.fft(truncated)
    unfiltered = filtered
    unfiltered = unfiltered[1:blockLength//2]
    decodedBits = [decode_constellation(z) for z in unfiltered]
    decodedBits = ''.join(decodedBits)
    return decodedBits


if __name__ == "__main__":
    y = load_audio_file(audio_path)
    decodedData = decode(y)
    print(decodedData)