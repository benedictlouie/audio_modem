import numpy as np
from tqdm import tqdm

blockLength = 1024
cyclicPrefix = 32

def load_file(file_path):
    return np.loadtxt(file_path)

def decodeConstellation(z):
    out = ''
    out += '0' if z.imag > 0 else '1'
    out += '0' if z.real > 0 else '1'
    return out

def binary_to_ascii(binary_string):
    binary_values = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
    ascii_string = ''.join([chr(int(bv, 2)) for bv in binary_values])
    return ascii_string

def write_bits_to_file(filename: str, bits: str, *, pad: bool = True) -> None:
    byte_values = (int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
    data = bytearray(byte_values)
    with open(filename, "wb") as f:
        f.write(data)

def decode(channel, received):
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
    start_index_of_raw_data = (len(file_name) + len(file_size) + 2) * 8
    raw_data = combinedBits[start_index_of_raw_data:]
    return file_name, raw_data


if __name__ == "__main__":
    channel = load_file("data/channel.csv")
    for i in tqdm(range(1, 10)):
        received = load_file(f"data/file{i}.csv")
        file_name, raw_data = decode(channel, received)
        file_name = f'output/{i}.{file_name.split('.')[-1]}'
        write_bits_to_file(file_name, raw_data)


    
