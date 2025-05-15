import numpy as np

def load_file(file_path):
    return np.loadtxt(file_path)


blockLength = 1024
cyclicPrefix = 32

if __name__ == "__main__":

    channel = load_file("data/channel.csv")
    received = load_file("data/file1.csv")

    extendedLength = blockLength + cyclicPrefix
    numBlocks = len(received) // extendedLength
    
    channel = np.array(channel)
    channel.resize(1024, refcheck=False)
    channelInFourier = np.fft.fft(channel)

    for i in range(numBlocks):
        truncated = received[i * extendedLength + cyclicPrefix : (i+1) * extendedLength]
        filtered = np.fft.fft(truncated)
        unfiltered = filtered / channelInFourier
        unfiltered = unfiltered[1:blockLength//2]
        print(unfiltered[:30])
        exit()