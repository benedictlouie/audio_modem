import numpy as np
from utils import *

def symbolsToSignal(symbols):
    ofdm_blocks = []
    for i in range(len(symbols) // SYMBOLS_PER_BLOCK):
        data = symbols[i * SYMBOLS_PER_BLOCK: (i+1) * SYMBOLS_PER_BLOCK]
        freq = np.zeros(N_DFT, dtype=complex)
        for i in range(1, FULL_SYMBOLS_PER_BLOCK+1):
            freq[i] = np.random.choice(list(MAPPING.values()))
        freq[HIGH_PASS_INDEX: LOW_PASS_INDEX] = data
        freq[-FULL_SYMBOLS_PER_BLOCK:] = np.conj(freq[1:FULL_SYMBOLS_PER_BLOCK+1][::-1])
        time = np.fft.ifft(freq)
        time = np.real(time)
        cyclic_prefix = time[-CYCLIC_PREFIX:]
        block_with_cp = np.concatenate([cyclic_prefix, time])
        ofdm_blocks.append(block_with_cp)
    return ofdm_blocks

def generateAudioFromBits(bits, path):

    symbols = bitsToSymbols(bits)
    ofdm_blocks = symbolsToSignal(symbols)

    pilot_bits = generate_pilot_bits()
    pilot_symbols = modulate(pilot_bits)
    pilot_blocks = symbolsToSignal(pilot_symbols)

    start_chirp = generate_chirp(CHIRP_LENGTH, SAMPLING_RATE, CHIRP_LOW, CHIRP_HIGH)
    end_chirp = generate_chirp(CHIRP_LENGTH, SAMPLING_RATE, CHIRP_HIGH, CHIRP_LOW)

    padding = np.zeros(BLOCK_LENGTH)

    signal = np.concatenate([
        padding,         # zero pad at the beginning
        start_chirp,     # start chirp
        *pilot_blocks,   # pilot blocks
        *ofdm_blocks,    # OFDM blocks
        end_chirp,       # end chirp
        padding          # zero pad at the end
    ])
    signal_normalized = signal / np.max(np.abs(signal))
    write(path, SAMPLING_RATE, signal_normalized.astype(np.float32))


if __name__ == "__main__":
    generateAudioFromBits(bits, DEFAULT_AUDIO_PATH)