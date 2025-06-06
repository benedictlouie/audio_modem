import os
import librosa
import numpy as np
from scipy.io.wavfile import write

from utils.parameters import *

def load_audio_file(file_path: str) -> np.ndarray:
    return librosa.load(file_path, sr=None)[0]

def write_wav(filename: str, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    data = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, sample_rate, data)

def get_non_repeating_bits(n: int, seed: int) -> np.ndarray:
    """
    Generate n bits from seed.
    """
    rng = np.random.default_rng(seed=seed)
    return rng.integers(0, 2, size=n)

def get_bitstream_from_symbols(symbols: np.ndarray, noise_variance) -> np.ndarray:
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
    LLR *= np.sqrt(2) / noise_variance

    # Stop at the last multiple of N
    LLR = LLR[:(len(LLR) // CODE.N) * CODE.N]

    # Decode LDPC
    num_max_iter = 0
    bitstream = np.array([])
    for i in range(0, len(LLR), CODE.N):
        # decoded is the final LLR after the message passing.
        # num_iterations is capped at 200.
        decoded, num_iterations = CODE.decode(LLR[i:i + CODE.N], DECTYPE)
        # Only the first K bits matter due to the systematic generator matrix 
        bitstream = np.concatenate((bitstream, decoded[:CODE.K]))
        num_max_iter += num_iterations == 200

    print(f"LDPC decoding finished with {num_max_iter} max iterations.")
    # Replace positive LLRs with 0 and negative LLRs with 1.
    bitstream = np.where(bitstream > 0, 0, 1)
    return bitstream, num_max_iter

def get_symbols_from_bitstream(bitstream: np.ndarray, skip_encoding: bool = False) -> np.ndarray:
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
        bitstream = np.concatenate((bitstream, np.random.default_rng(42).integers(2, size=(-len(bitstream)) % CODE.K)))

        # Encode LDPC
        encoded_bitstream = np.array([])
        for i in range(0, len(bitstream), CODE.K):
            encoded_bitstream = np.concatenate((encoded_bitstream, CODE.encode(bitstream[i:i + CODE.K])))

    # Turn every two bits into a QPSK symbol [1+j, 1-j, -1-j, -1+j]
    encoded_bitstream = np.where(encoded_bitstream == 0, 1, -1)
    symbols = encoded_bitstream[::2] + 1j * encoded_bitstream[1::2]
    return symbols

def get_known_blocks(num_frames: int) -> np.ndarray:
    """
    Generate known blocks of symbols for synchronization.
    Returns a matrix with BLOCK_LENGTH columns.
    """
    symbols = get_symbols_from_bitstream(get_non_repeating_bits(BITS_PER_SYMBOL * EFFECTIVE_SYMBOLS_PER_BLOCK * num_frames, 42), skip_encoding=True)
    symbols = symbols.reshape(-1, EFFECTIVE_SYMBOLS_PER_BLOCK)
    blocks = np.fft.ifft(np.concatenate((
        np.zeros((num_frames, 1)),
        symbols,
        np.zeros((num_frames, 1)),
        np.conjugate(symbols[:, ::-1]),
    ), axis=1), axis=1).real

    blocks = np.concatenate((blocks[:, -CYCLIC_PREFIX:], blocks), axis=1)

    return blocks

def get_chirp() -> np.ndarray:
    """
    Generate a chirp signal.
    """
    t = np.linspace(0, CHIRP_TIME, CHIRP_LENGTH)
    signal = CHIRP_FACTOR * np.sin(np.pi * (CHIRP_LOW + (CHIRP_HIGH - CHIRP_LOW) * t / CHIRP_TIME) * t)
    return signal

def text_to_binary(text: str) -> str:
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_str: str) -> str:
    return ''.join(chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8))

def encode_file_to_bits(input_filepath: str) -> np.ndarray:
    filename = os.path.basename(input_filepath)
    with open(input_filepath, "rb") as f:
        data = f.read()
    bit_size = len(data)

    bits_filename = np.unpackbits(np.frombuffer(filename.encode('utf-8'), dtype=np.uint8))
    bits_null = np.unpackbits(np.frombuffer(b'\x00', dtype=np.uint8))
    bits_bit_size = np.unpackbits(np.frombuffer(str(bit_size).encode('utf-8'), dtype=np.uint8))
    bits_data = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    # Concatenate all parts
    full_bits = np.concatenate([
        bits_filename, bits_null,
        bits_bit_size, bits_null,
        bits_data
    ])

    return full_bits.astype(np.uint8)

def decode_bits_to_file(bits: np.ndarray, output_dir: str = "."):
    byte_data = np.packbits(bits).tobytes()
    
    # Split byte data at null bytes (up to 2 times)
    parts = byte_data.split(b'\x00', 2)
    if len(parts) < 3:
        raise ValueError("Invalid encoded format")

    filename_bytes, bit_size_bytes, file_data = parts

    try:
        filename = filename_bytes.decode('utf-8')
        bit_size = int(bit_size_bytes.decode('utf-8'))
        file_data = file_data[:bit_size]
    except Exception as e:
        raise ValueError("Failed to parse metadata") from e

    output_path = os.path.join(output_dir, "received_" + filename)
    with open(output_path, "wb") as f:
        f.write(file_data)

    print(f"Decoded file written to: {output_path}")
    return output_path

def get_original_bits() -> np.ndarray:
    return encode_file_to_bits(FILE_PATH)