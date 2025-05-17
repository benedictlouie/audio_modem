import numpy as np
from typing import Tuple

from utils import *

# Doing OFDM with every block
def encode_block(symbols: np.ndarray) -> np.ndarray:

    # Start with one 0, add all the symbols, another 0 then conjugate the symbols
    symbols = np.concatenate((np.zeros(1), symbols, np.zeros(1), np.conjugate(symbols[::-1])))

    # IFFT
    symbolsInTime = np.fft.ifft(symbols).real

    # Add the cyclic prefix in front
    return np.concatenate((symbolsInTime[-cyclicPrefix:], symbolsInTime))

# Encode the bitstream
def encode(bitstream: str, syncBlock: bool=False) -> np.ndarray:

    # Get every two bits from the bitstream and turn into constellation
    symbols = np.array([constellation[bitstream[i:i+2]] for i in range(0, len(bitstream), 2)])

    if not syncBlock:
        # Repeat symbols and make sure we have syncLength symbols
        symbols = np.repeat(symbols, repeatCount)
        symbols = np.concatenate((symbols, np.repeat(constellation['00'], (-len(symbols)) % syncLength)))
        

    # We need this many DFT blocks
    blockCount = len(symbols) // symbolsPerBlock

    # Initialise an all-zero signal array
    signal = np.zeros(blockCount * blockLength)

    # For every block, we encode the signal
    for i in range(blockCount):
        signal[i * blockLength : (i+1) * blockLength] = encode_block(symbols[i * symbolsPerBlock : (i+1) * symbolsPerBlock])

    return signal

def insert_sync_blocks(signal: np.ndarray) -> np.ndarray:

    startBlock, syncBlock, endBlock = get_sync_blocks()
    syncLength = syncBlockPeriod * blockLength

    output = np.array([])
    for i in range(0, len(signal), syncLength):
        output = np.concatenate((output, signal[i: i + syncLength], syncBlock))
    output = np.concatenate((np.zeros(sampleRate),
                             startBlock,
                             output[:-blockLength], # remove the last sync block
                             endBlock,
                             np.zeros(sampleRate)))
    return output

def get_sync_blocks() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bits = get_non_repeating_bits(2 * symbolsPerBlock * (2 * startEndBlockMultiplier + 1))
    blocks = encode(bits, syncBlock=True)

    startBlock = blocks[:blockLength * startEndBlockMultiplier]
    syncBlock = blocks[blockLength * startEndBlockMultiplier: blockLength * (startEndBlockMultiplier + 1)]
    endBlock = blocks[blockLength * (startEndBlockMultiplier + 1):]

    return startBlock, syncBlock, endBlock
    
if __name__ == "__main__":
    text = """
In Cambridge's halls where knowledge flows,
A beacon of wisdom, his presence shows.
From Zürich's peaks to England's plains,
He charts the course where learning reigns.

So here's to Sayir, whose endless quest,
Ignites the minds, inspires the best.
A luminary in academia's sphere,
His legacy shines, year after year.
"""
    data = text_to_binary(text)
    # data = encode_ldpc(data)
    signal = encode(data)
    signal = insert_sync_blocks(signal)
    write_wav(audio_path, signal)
    print(f'Bitrate: {round(len(data) * sampleRate / (len(signal) - 2 * sampleRate))} bps')