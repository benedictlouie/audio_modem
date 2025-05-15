import numpy as np
from PIL import Image
import wave
import struct

def load_file(file_path):
    return np.loadtxt(file_path)

blockLength = 1024
cyclicPrefix = 32

def decodeConstellation(z):
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

def binary_to_ascii(binary_string):
    binary_values = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
    ascii_string = ''.join([chr(int(bv, 2)) for bv in binary_values])
    return ascii_string

def create_wav_from_bits(bits, filename, sample_rate=44100, num_channels=1, sample_width=1):
    num_samples = len(bits) // 8  # each sample is 8 bits
    audio_data = []
    for i in range(num_samples):
        sample = int(bits[i*8:(i+1)*8], 2)  # Get each 8-bit chunk
        audio_data.append(sample)
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)  # Mono or Stereo
        wav_file.setsampwidth(sample_width)  # Sample width in bytes (2 for 16-bit)
        wav_file.setframerate(sample_rate)  # Sample rate (44100 Hz is typical for audio)
        wav_file.writeframes(struct.pack('<' + 'h' * len(audio_data), *audio_data))


if __name__ == "__main__":


    channel = load_file("data/channel.csv")
    received = load_file("data/file2.csv")

    extendedLength = blockLength + cyclicPrefix
    numBlocks = len(received) // extendedLength

    channel = np.array(channel)
    channel.resize(blockLength, refcheck=False)
    channelInFourier = np.fft.fft(channel)

    combinedBits = ""
    for i in range(numBlocks):
        truncated = received[i * extendedLength + cyclicPrefix : (i+1) * extendedLength]
        filtered = np.fft.fft(truncated)
        unfiltered = filtered / channelInFourier
        unfiltered = unfiltered[1:blockLength//2]
        decodedBits = [decodeConstellation(z) for z in unfiltered]
        decodedBits = ''.join(decodedBits)
        combinedBits += decodedBits
    decoded_string = binary_to_ascii(combinedBits)
    file_name, file_size, _ = decoded_string.split("\0", 2)
    print(file_name, file_size)

    start_index_of_raw_data = (len(file_name) + len(file_size) + 2) * 8
    raw_data = combinedBits[start_index_of_raw_data:]
    print(binary_to_ascii(raw_data)[:44])
    print(len(raw_data))
    create_wav_from_bits(raw_data, "output.wav", 16000)
