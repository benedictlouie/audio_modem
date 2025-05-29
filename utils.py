import contextlib
import ldpc
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

SAMPLE_RATE = 48000

# Number of symbols [1:N_DFT//2] per block
EFFECTIVE_SYMBOLS_PER_BLOCK = 2 ** 12 - 1
CYCLIC_PREFIX = 2 ** 11
BLOCK_LENGTH = 2 * (EFFECTIVE_SYMBOLS_PER_BLOCK + 1) + CYCLIC_PREFIX
N_DFT = BLOCK_LENGTH - CYCLIC_PREFIX

# Cut-offs because of hardware limitation
LOW_PASS_INDEX = round(0.9 * EFFECTIVE_SYMBOLS_PER_BLOCK)
HIGH_PASS_INDEX = round(0.01 * EFFECTIVE_SYMBOLS_PER_BLOCK)

# Number of symbols after cut-offs
SYMBOLS_PER_BLOCK = LOW_PASS_INDEX - HIGH_PASS_INDEX

WIENER_SNR = 10

# number of information blocks following each known block
INFORMATION_BLOCKS_PER_FRAME = 4

# Number of blocks sent after LDPC
INFORMATION_BLOCKS = 20

CHIRP_TIME = 0.5
CHIRP_LENGTH = round(CHIRP_TIME * SAMPLE_RATE)
CHIRP_FACTOR = 0.1
CHIRP_LOW = 0
CHIRP_HIGH = 5000

# QPSK
BITS_PER_SYMBOL = 2
assert BITS_PER_SYMBOL == 2

AUDIO_PATH = "output.wav"

# LDPC Settings
DECTYPE = 'sumprod2'
CODE = ldpc.code()

def load_audio_file(file_path: str) -> np.ndarray:
    with contextlib.redirect_stderr(None):
        return librosa.load(file_path, sr=None)[0]

def write_wav(filename: str, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def get_non_repeating_bits(n: int, seed: int) -> str:
    """
    Generate n bits from seed.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 2, size=n)

def get_bitstream_from_symbols(symbols: np.ndarray, noise_variance) -> str:
    """
    FOR DECODING
    Convert symbols to a bitstream using the QPSK mapping.
    Perform LDPC decoding.
    """

    # Use the real and imaginary parts of the symbols to find LLR.
    LLR = np.empty(len(symbols) * BITS_PER_SYMBOL)
    LLR[::2] = symbols.real
    LLR[1::2] = symbols.imag
    
    # Find LLR
    if not isinstance(noise_variance, (np.ndarray, list, tuple)): noise_variance = np.array([noise_variance])
    assert len(LLR) % len(noise_variance) == 0
    noise_variance = np.tile(noise_variance, len(LLR) // len(noise_variance))
    LLR *= np.sqrt(2) * np.sqrt(2) / noise_variance

    # Stop at the last multiple of N
    LLR = LLR[:(len(LLR) // CODE.N) * CODE.N]
    # encoded_bitstream = unpermute(encoded_bitstream)

    # Decode LDPC
    bitstream = np.array([])
    divergeCount = 0
    for i in range(0, len(LLR), CODE.N):
        # decoded is the final LLR after the message passing.
        # num_iterations is capped at 200.
        decoded, num_iterations = CODE.decode(LLR[i:i + CODE.N], DECTYPE)
        if num_iterations >= 200: divergeCount += 1
        bitstream = np.concatenate((bitstream, decoded[:CODE.K]))
    print("Diverged:", divergeCount, "/", len(LLR)//CODE.N)

    # Replace positive LLRs with 0 and negative LLRs with 1.
    bitstream = np.where(bitstream > 0, 0, 1)
    return bitstream

def get_symbols_from_bitstream(bitstream: str, skip_encoding: bool = False) -> np.ndarray:
    """
    FOR ENCODING
    Convert a bitstream to symbols using the constellation mapping.
    Perform LDPC encoding.
    """
    
    if skip_encoding:
        # Skip encoding if we are transforing pilot bits.
        encoded_bitstream = bitstream
    else:
        # Pad until a multiple of CODE.K before doing LDPC
        bitstream = np.concatenate((bitstream, np.random.randint(2, size=(-len(bitstream)) % CODE.K)))

        # Encode LDPC
        encoded_bitstream = np.array([])
        for i in range(0, len(bitstream), CODE.K):
            encoded_bitstream = np.concatenate((encoded_bitstream, CODE.encode(bitstream[i:i + CODE.K])))
        # encoded_bitstream[:] = permute(encoded_bitstream)

    # Turn every two bits into a QPSK symbol [1+j, 1-j, -1-j, -1+j]
    encoded_bitstream = np.where(encoded_bitstream == 0, 1, -1)
    symbols = encoded_bitstream[::2] + 1j * encoded_bitstream[1::2]
    return symbols

def plot_sent_received_constellation(sent: np.ndarray, received: np.ndarray) -> None:
    """
    Plot the constellation of sent and received symbols.
    Sent symbols are used to color received ones, and ideal sent locations are also shown.
    """

    # received = np.reshape(received, (-1, SYMBOLS_PER_BLOCK))[:,SYMBOLS_PER_BLOCK*9//10:SYMBOLS_PER_BLOCK].flatten()
    # sent = np.reshape(sent, (-1, SYMBOLS_PER_BLOCK))[:,SYMBOLS_PER_BLOCK*9//10:SYMBOLS_PER_BLOCK].flatten()

    unique_symbols = np.unique(sent)
    colors = ['red', 'blue', 'green', 'orange']
    color_map = dict(zip(unique_symbols, colors))

    plt.figure(figsize=(8, 8))

    # Received symbols marked as dots
    for sym in unique_symbols:
        mask = sent == sym
        plt.scatter(received[mask].real, received[mask].imag,
                    color=color_map[sym], alpha=0.1, label=f'Received (Sent: {sym})')

    # Sent symbols marked with an "x"
    for sym in unique_symbols:
        plt.plot(sym.real, sym.imag, 'x', markersize=12, markeredgewidth=2,
                 color=color_map[sym], label=f'Sent: {sym}')

    accuracy = np.mean(sent == np.sign(received.real) + 1j * np.sign(received.imag))

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title(f'Constellation: Sent vs Received. Accuracy: {accuracy:.4f}.')
    plt.axis('equal')
    plt.show()

def plot_error_per_bin(received: np.ndarray, sent: np.ndarray, filter: np.ndarray) -> None:
    """
    Plot the filter magnitude (in oragne) and the bit error rate (in blue) after applying the filter.
    """

    received = np.reshape(received, (-1, SYMBOLS_PER_BLOCK))
    sent = np.reshape(sent, (-1, SYMBOLS_PER_BLOCK))

    received = np.sign(received.real) + 1j * np.sign(received.imag)
    error_rate = np.mean(received != sent, axis=0)
    filter_magnitude = np.abs(np.mean(filter, axis=0))[1: SYMBOLS_PER_BLOCK + 1]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.bar(np.arange(SYMBOLS_PER_BLOCK), error_rate, color=color, alpha=0.6, label='Error Rate')
    ax1.set_xlabel('Bin Index')
    ax1.set_ylabel('Error Rate', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.plot(np.arange(SYMBOLS_PER_BLOCK), filter_magnitude, color=color, label='Filter Magnitude')
    ax2.set_ylabel('Filter Magnitude', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Error Rate per Bin and Filter Magnitude')
    fig.tight_layout()
    plt.show()

def permute(arr: np.ndarray):
    perm = np.random.default_rng(seed=42).permutation(len(arr))
    return arr[perm]

def unpermute(shuffled: np.ndarray):
    perm = np.random.default_rng(seed=42).permutation(len(shuffled))
    rev_perm = np.argsort(perm)
    return shuffled[rev_perm]

def text_to_binary(text: str) -> str:
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_str: str) -> str:
    return ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))

POEM = """In Cambridge's halls where knowledge flows,
A beacon of wisdom, his presence shows.
From Zürich's peaks to England's plains,
He charts the course where learning reigns.

With circuits, codes, and signals bright,
He deciphers truths, brings them to light.
In lectures filled with passion's fire,
He lifts young minds, inspires higher.

Through channels where data streams align,
He weaves the threads, designs the sign.
A mentor, guide, and scholar true,
In every task, excellence he pursues.

Awards may grace his learned name,
Yet humble hearts define his fame.
In every student's grateful voice,
Echoes the impact of his choice.

So here's to Sayir, whose endless quest,
Ignites the minds, inspires the best.
A luminary in academia's sphere,
His legacy shines, year after year.
"""

# Bits to be sent
# Recall INFORMATION_BLOCKS is the number of blocks sent after LDPC.
DATA = get_non_repeating_bits(SYMBOLS_PER_BLOCK * BITS_PER_SYMBOL * INFORMATION_BLOCKS // CODE.N * CODE.K, 69)