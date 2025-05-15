import numpy as np

def load_file(file_path):
    return np.loadtxt(file_path)

blockLength = 1024
cyclicPrefix = 32

def decodeConstellation(z):
    if z.real > 0:
        if z.imag >= 0: return "00"
        else: return "01"
    else:
        if z.imag >= 0: return "10"
        else: return "11"

def demodulate(binary_data):
    # Convert binary string to a byte array
    byte_data = bytearray(int(binary_data[i:i+8], 2) for i in range(0, len(binary_data), 8))
    
    # Find the first two null-terminators ('\0')
    first_null_idx = byte_data.index(0)  # First '\0' for filename
    second_null_idx = byte_data.index(0, first_null_idx + 1)  # Second '\0' for filesize
    
    # Extract the filename (a string before the first '\0')
    filename = byte_data[:first_null_idx].decode('utf-8', errors='ignore')
    
    # Extract the file size (a string between the first and second '\0')
    file_size_str = byte_data[first_null_idx + 1:second_null_idx].decode('utf-8')
    file_size = int(file_size_str)
    
    # Extract the raw file data (starting right after the second '\0' till the end)
    raw_data = byte_data[second_null_idx + 1:second_null_idx + 1 + file_size]
    
    # Write the raw data to a file
    with open(filename, 'wb') as file:
        file.write(raw_data)
    
    return filename, file_size, raw_data

if __name__ == "__main__":

    channel = load_file("data/channel.csv")
    received = load_file("data/file1.csv")

    extendedLength = blockLength + cyclicPrefix
    numBlocks = len(received) // extendedLength
    
    channel = np.array(channel)
    channel.resize(blockLength, refcheck=False)
    channelInFourier = np.fft.fft(channel)

    for i in range(numBlocks):
        truncated = received[i * extendedLength + cyclicPrefix : (i+1) * extendedLength]
        filtered = np.fft.fft(truncated)
        unfiltered = filtered / channelInFourier
        unfiltered = unfiltered[1:blockLength//2]
        decodedBits = [decodeConstellation(z) for z in unfiltered]
        decodedBits = ''.join(decodedBits)
        filename, file_size, raw_data = demodulate(decodedBits)
        print(filename)
        print(file_size)
        print(raw_data.decode('utf-8'))
        exit()