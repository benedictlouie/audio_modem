import contextlib
import ldpc
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

SAMPLE_RATE = 48000
SYMBOLS_PER_BLOCK = 2 ** 12 - 1
CYCLIC_PREFIX = 2 ** 10
BLOCK_LENGTH = 2 * (SYMBOLS_PER_BLOCK + 1) + CYCLIC_PREFIX

ESTIMATE_CHANNEL_BLOCKS = 10
INFORMATION_BLOCKS = 1
BLOCKS_PER_SYNC = 1

WIENER_SNR = 1

START_END_CHIRP_TIME = 0.4
SYNC_CHIRP_TIME = 0.2
START_END_CHIRP_LENGTH = round(START_END_CHIRP_TIME * SAMPLE_RATE)
SYNC_CHIRP_LENGTH = round(SYNC_CHIRP_TIME * SAMPLE_RATE)
CHIRP_FACTOR = 0.04
CHIRP_LOW = 0
CHIRP_HIGH = 5000


BITS_PER_CONSTELLATION = 2

AUDIO_PATH = "output.wav"

DECTYPE = 'sumprod2'
CODE = ldpc.code()

def load_audio_file(file_path: str) -> np.ndarray:
    with contextlib.redirect_stderr(None):
        return librosa.load(file_path, sr=None)[0]

def write_wav(filename: str, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def get_non_repeating_bits(n: int, seed: int) -> str:
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 2, size=n)

def get_bitstream_from_symbols(symbols: np.ndarray) -> str:
    """
    Convert symbols to a bitstream using the constellation mapping.
    """
    encoded_bitstream = np.empty(len(symbols) * BITS_PER_CONSTELLATION)
    encoded_bitstream[::2] = symbols.real
    encoded_bitstream[1::2] = symbols.imag
    
    encoded_bitstream = encoded_bitstream[:(len(encoded_bitstream) // CODE.N) * CODE.N]
    bitstream = np.array([])
    for i in range(0, len(encoded_bitstream), CODE.N):
        bitstream = np.concatenate((bitstream, CODE.decode(encoded_bitstream[i:i + CODE.N], DECTYPE)[0][:CODE.K]))

    bitstream = np.where(bitstream > 0, 0, 1)
    return bitstream
    

def get_symbols_from_bitstream(bitstream: str, skip_encoding: bool = False) -> np.ndarray:
    """
    Convert a bitstream to symbols using the constellation mapping.
    """
    if skip_encoding:
        encoded_bitstream = bitstream
    else:
        bitstream = np.concatenate((bitstream, np.zeros((-len(bitstream)) % CODE.K)))
        encoded_bitstream = np.array([])
        for i in range(0, len(bitstream), CODE.K):
            encoded_bitstream = np.concatenate((encoded_bitstream, CODE.encode(bitstream[i:i + CODE.K])))

    encoded_bitstream = np.where(encoded_bitstream == 0, 1, -1)
    symbols = encoded_bitstream[::2] + 1j * encoded_bitstream[1::2]
    return symbols

def plot_sent_received_constellation(sent: np.ndarray, received: np.ndarray) -> None:
    """
    Plot the constellation of sent and received symbols.
    Sent symbols are used to color received ones, and ideal sent locations are also shown.
    """

    unique_symbols = np.unique(sent)
    colors = ['red', 'blue', 'green', 'orange']
    color_map = dict(zip(unique_symbols, colors))

    plt.figure(figsize=(8, 8))

    for sym in unique_symbols:
        mask = sent == sym
        plt.scatter(received[mask].real, received[mask].imag,
                    color=color_map[sym], alpha=0.6, label=f'Received (Sent: {sym})')

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

ESTIMATE_CHANNEL_DATA = get_non_repeating_bits(SYMBOLS_PER_BLOCK * BITS_PER_CONSTELLATION * ESTIMATE_CHANNEL_BLOCKS, 1)
DATA = get_non_repeating_bits(SYMBOLS_PER_BLOCK * BITS_PER_CONSTELLATION * INFORMATION_BLOCKS // 648 * 324, 2)