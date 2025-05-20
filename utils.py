import contextlib
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pyldpc
from scipy.io.wavfile import write
from tqdm import tqdm

SAMPLE_RATE = 48000
SYMBOLS_PER_BLOCK = 511
CYCLIC_PREFIX = 512
BLOCK_LENGTH = 2 * (SYMBOLS_PER_BLOCK + 1) + CYCLIC_PREFIX

CHIRP_TIME = 0.5
CHIRP_FACTOR = 0.1
CHIRP_LENGTH = round(CHIRP_TIME * SAMPLE_RATE)
CHIRP_LOW = 0
CHIRP_HIGH = 5000

CONSTELLATION = {
    '00': 0.707 + 0.707j,
    '01': -0.707 + 0.707j,
    '10': 0.707 - 0.707j,
    '11': -0.707 - 0.707j
}
BITS_PER_CONSTELLATION = int(np.log2(len(CONSTELLATION)))

REV_CONSTELLATION = {coord: bits for bits, coord in CONSTELLATION.items()}

AUDIO_PATH = "output.wav"

N = 15
D_V = 4
D_C = 5
SNR = 0.1
H, G = pyldpc.make_ldpc(N, D_V, D_C, sparse=True, seed=0)
K = G.shape[1]

def load_audio_file(file_path: str) -> np.ndarray:
    with contextlib.redirect_stderr(None):
        return librosa.load(file_path, sr=None)[0]

def write_wav(filename: str, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def get_non_repeating_bits(n: int, seed: int) -> str:
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 2, size=n)

def get_white_noise(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(0, 1, n)
    return noise

def decode_constellation(z: complex) -> str:
    """
    Decode a complex number to its corresponding bit representation.
    """
    closestConstellation = None
    minDist = float('inf')
    for constellation in REV_CONSTELLATION:
        dist = np.abs(z - constellation)
        if dist < minDist:
            minDist = dist
            closestConstellation = constellation
    return REV_CONSTELLATION[closestConstellation]

def get_bitstream_from_symbols(symbols: np.ndarray) -> str:
    """
    Convert symbols to a bitstream using the constellation mapping.
    """
    encoded_bitstream = np.empty(len(symbols) * BITS_PER_CONSTELLATION)
    encoded_bitstream[::2] = symbols.real
    encoded_bitstream[1::2] = symbols.imag
    
    encoded_bitstream = encoded_bitstream[:(len(encoded_bitstream) // N) * N]
    bitstream = np.array([])
    for i in tqdm(range(0, len(encoded_bitstream), N)):
        bitstream = np.concatenate((bitstream, pyldpc.get_message(G, pyldpc.decode(H, encoded_bitstream[i:i + N], SNR))))
    return bitstream
    

def get_symbols_from_bitstream(bitstream: str) -> np.ndarray:
    """
    Convert a bitstream to symbols using the constellation mapping.
    """
    bitstream = np.concatenate((bitstream, np.zeros((-len(bitstream)) % K)))

    encoded_bitstream = np.array([])
    for i in range(0, len(bitstream), K):
        encoded_bitstream = np.concatenate((encoded_bitstream, pyldpc.encode(G, bitstream[i:i + K], float('inf'))))
    
    symbols = np.round(encoded_bitstream[::2]) + 1j * np.round(encoded_bitstream[1::2])
    return symbols

def plot_sent_received_constellation(sent: np.ndarray, received: np.ndarray) -> None:
    """
    Plot the constellation of sent and received symbols.
    Sent symbols are used to color received ones, and ideal sent locations are also shown.
    """

    unique_symbols = np.unique(sent)

    # Assign a color to each type
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    color_map = dict(zip(unique_symbols, colors))

    # Prepare plot
    plt.figure(figsize=(8, 8))

    # Plot received symbols, colored by sent symbol type
    for sym in unique_symbols:
        mask = sent == sym
        plt.scatter(received[mask].real, received[mask].imag,
                    color=color_map[sym], alpha=0.6, label=f'Received (Sent: {sym})')
        
    # Overlay the ideal sent symbols
    for sym in unique_symbols:
        plt.plot(sym.real, sym.imag, 'x', markersize=12, markeredgewidth=2,
                 color=color_map[sym], label=f'Sent: {sym}')

    # Formatting
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title(f'Constellation: Sent vs Received')
    # plt.legend()
    plt.axis('equal')  # Preserve aspect ratio
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

DATA = get_non_repeating_bits(SYMBOLS_PER_BLOCK * BITS_PER_CONSTELLATION * 100, 69)