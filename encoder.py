import numpy as np

from utils import *

# Doing OFDM with every block
def encode_block(symbols):

    # Start with one 0, add all the symbols, another 0 then conjugate the symbols
    symbols = np.concatenate((np.zeros(1), symbols, np.zeros(1), np.conjugate(symbols[::-1])))

    # IFFT
    symbolsInTime = np.fft.ifft(symbols).real

    # Add the cyclic prefix in front
    return np.concatenate((symbolsInTime[-cyclicPrefix:], symbolsInTime))

# Encode the bitstream
def encode(bitstream, repeat=True):

    # Get every two bits from the bitstream and turn into constellation
    symbols = np.array([constellation[bitstream[i:i+2]] for i in range(0, len(bitstream), 2)])

    # Repetition coding
    if repeat:
        symbols = np.repeat(symbols, sampleRate // symbolRate)

    # Pad the 00 constellation if we have remainder
    remainder = len(symbols) % symbolsPerBlock
    if remainder:
        symbols = np.concatenate((symbols, np.repeat(constellation['00'], symbolsPerBlock - remainder)))

    # We need this many DFT blocks
    blockCount = len(symbols) // symbolsPerBlock

    # Initialise an all-zero signal array
    signal = np.zeros(blockCount * blockLength)

    # For every block, we encode the signal
    for i in range(blockCount):
        signal[i * blockLength : (i+1) * blockLength] = encode_block(symbols[i * symbolsPerBlock : (i+1) * symbolsPerBlock])

    return signal

def insert_sync_blocks(signal):

    syncBlockCount = len(signal) // blockLength // syncBlockPeriod
    syncBlocks = synchronize_blocks(syncBlockCount + 2)
    output = np.zeros(sampleRate)
    for i in range(syncBlockCount):
        output = np.concatenate((output, syncBlocks[i+1], signal[i * syncBlockPeriod * blockLength : (i+1) * syncBlockPeriod * blockLength]))
    output = np.concatenate((output, syncBlocks[0], syncBlocks[syncBlockCount + 1], np.zeros(sampleRate)))
    return output

# Generate blockCount synchronisation blocks
def synchronize_blocks(blockCount):
    # Multiply by 2 because we have 2 bits per symbol
    sequence = encode(get_non_repeating_bits(2 * blockCount * symbolsPerBlock), repeat=False)
    return [sequence[i * blockLength: (i+1) * blockLength] for i in range(blockCount)]
    
if __name__ == "__main__":
    data = text_to_binary("My name is Ben, I am a software engineer.")
    signal = encode(data)
    signal = insert_sync_blocks(signal)
    write_wav(audio_path, signal)