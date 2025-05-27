import numpy as np
import matplotlib.pyplot as plt
from utils import *

def getStartEndIndex(received_signal):
    start_chirp = generate_chirp(CHIRP_LENGTH, SAMPLING_RATE, CHIRP_LOW, CHIRP_HIGH)
    end_chirp = generate_chirp(CHIRP_LENGTH, SAMPLING_RATE, CHIRP_HIGH, CHIRP_LOW)
    corr_start = correlate(received_signal, start_chirp, mode='valid')
    start_index = np.argmax(np.abs(corr_start))
    corr_end = correlate(received_signal, end_chirp, mode='valid')
    end_index = np.argmax(np.abs(corr_end))
    return start_index, end_index

def resampleUsingStartAndEnd(signal):
    start_index, end_index = getStartEndIndex(signal)
    extracted_ofdm = signal[start_index:end_index]

    NUMBER_OF_BLOCKS_WITH_CODING = round((len(extracted_ofdm) - CHIRP_LENGTH) / BLOCK_LENGTH) - NUMBER_OF_PILOT_BLOCKS
    expected_length = CHIRP_LENGTH + BLOCK_LENGTH * (NUMBER_OF_PILOT_BLOCKS + NUMBER_OF_BLOCKS_WITH_CODING)

    sampling_ratio = expected_length / len(extracted_ofdm)
    extracted_resampled = librosa.resample(extracted_ofdm, orig_sr=SAMPLING_RATE, target_sr=SAMPLING_RATE * sampling_ratio)
    extracted_resampled = extracted_resampled[CHIRP_LENGTH: CHIRP_LENGTH+expected_length]
    return extracted_resampled

def signalToSymbols(signal):
    symbols = []
    for block in signal:
        block_no_cp = block[CYCLIC_PREFIX:]           # Remove CP → now length = N_DFT
        block_fft = np.fft.fft(block_no_cp)           # FFT → returns N_DFT complex values
        symbols.append(block_fft)
    symbols = np.array(symbols)
    return symbols

def estimate_channel_coefficients_from_pilot_symbols(received_pilot_symbols):
    pilot_bits = generate_pilot_bits()
    pilot_symbols = modulate(pilot_bits)
    pilot_symbols = pilot_symbols.reshape(received_pilot_symbols.shape)
    estimated_channel_coefficients = received_pilot_symbols / pilot_symbols
    estimated_channel_coefficients = np.mean(estimated_channel_coefficients, axis=0)
    return estimated_channel_coefficients    

def get_filter(channel_coefficients, shape, SNR=SNR):
    filter = np.conjugate(channel_coefficients)/(np.abs(channel_coefficients)**2 + 1/SNR)
    big_filter = np.empty(shape, dtype=complex)
    for i in range(len(big_filter)):
        big_filter[i,:] = filter
    return big_filter

def recoverSymbolsFromAudio(audio):
    received_signal, sr = librosa.load(audio, sr=SAMPLING_RATE, mono=True)
    resampled_signal = resampleUsingStartAndEnd(received_signal)

    ofdm_blocks_time = resampled_signal.reshape((-1, BLOCK_LENGTH))
    ofdm_blocks_freq = signalToSymbols(ofdm_blocks_time)

    demodulated_pilot_symbols = ofdm_blocks_freq[:NUMBER_OF_PILOT_BLOCKS, HIGH_PASS_INDEX: LOW_PASS_INDEX]
    demodulated_symbols = ofdm_blocks_freq[NUMBER_OF_PILOT_BLOCKS:, HIGH_PASS_INDEX: LOW_PASS_INDEX]
    estimated_channel_coefficients = estimate_channel_coefficients_from_pilot_symbols(demodulated_pilot_symbols)
    filter = get_filter(estimated_channel_coefficients, demodulated_symbols.shape)
    
    recovered_symbols = demodulated_symbols * filter
    return recovered_symbols

constellation_points = list(MAPPING.values())

def find_closest_constellation(symbol, constellation):
    distances = [np.abs(symbol - c) for c in constellation]
    return np.argmin(distances)

def plotConstellationDiagram(sent_symbols, recovered_symbols):

    colors = ['red', 'blue', 'green', 'purple']

    tx = sent_symbols.flatten()
    rx = recovered_symbols.flatten()
    rx = rx / np.sqrt(np.mean(np.abs(rx)**2))

    # Get color labels based on transmitted symbols' closest constellation points
    color_labels = np.array([find_closest_constellation(sym, constellation_points) for sym in tx])

    # Plot received points colored by intended QPSK cluster
    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.scatter(rx.real[color_labels == i], rx.imag[color_labels == i], color=colors[i], label=f'Cluster {i}', alpha=0.1)

    # Plot ideal constellation points in matching color
    for i, point in enumerate(constellation_points):
        plt.plot(point.real, point.imag, 'x', color=colors[i], markersize=12, mew=2, label=f'QPSK {i}')

    # Formatting
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axis('equal')

    decoded_symbols = np.array([find_closest_constellation(sym, constellation_points) for sym in rx])
    original_symbols = np.array([find_closest_constellation(sym, constellation_points) for sym in tx])
    correctness = np.sum(decoded_symbols == original_symbols) / len(original_symbols)
    plt.title(f"Constellation Diagram (Correctness = {correctness})")
    plt.show()

def QPSKBitErrorRatePreLDPC(recovered_symbols, encoded_bitstream):
    
    assert BITS_PER_SYMBOL == 2
    rx = recovered_symbols.flatten()
    rx = rx / np.sqrt(np.mean(np.abs(rx)**2))

    decoded_symbols = np.array([find_closest_constellation(sym, constellation_points) for sym in rx])

    decoded_bits = ""
    for sym in decoded_symbols:
        if sym == 0: decoded_bits += "00"
        elif sym == 1: decoded_bits += "01"
        elif sym == 2: decoded_bits += "10"
        elif sym == 3: decoded_bits += "11"
        else: exit("symbol error")
    decoded_bits = np.array([int(bit) for bit in decoded_bits])
    decoded_bits = decoded_bits[:len(encoded_bitstream)]
    return np.sum(decoded_bits != encoded_bitstream) / len(encoded_bitstream)

def decodeLDPCFromQPSK(recovered_symbols):

    assert BITS_PER_SYMBOL == 2
    rx = recovered_symbols.flatten()
    rx = rx / np.sqrt(np.mean(np.abs(rx)**2))

    LLR = np.array([[sym.real, sym.imag] for sym in rx]).flatten()
    LLR = LLR[: len(LLR) // CODE.N * CODE.N]
    bitstream = np.array([])
    for i in range(0, len(LLR), CODE.N):
        bitstream = np.concatenate((bitstream, CODE.decode(LLR[i:i + CODE.N], DECTYPE)[0][:CODE.K]))
    bitstream = np.where(bitstream > 0, 0, 1)
    bitstream = bitstream[:len(bits)]
    return bitstream

def QPSKBitErrorRateAfterLDPC(recovered_bits, original_bits):
    return np.sum(recovered_bits != original_bits) / len(original_bits)
    
if __name__ == "__main__":
    received_audio = DEFAULT_AUDIO_PATH
    received_audio = "Untitled.aifc" # your file

    recovered_symbols = recoverSymbolsFromAudio(received_audio)
    sent_symbols = bitsToSymbols(bits)

    plotConstellationDiagram(sent_symbols, recovered_symbols)
    errorPreLDPC = QPSKBitErrorRatePreLDPC(recovered_symbols, encodeLDPC(bits))
    print("Bit error rate pre-LDPC:", errorPreLDPC)

    recovered_bits = decodeLDPCFromQPSK(recovered_symbols)
    errorPostLDPC = QPSKBitErrorRateAfterLDPC(recovered_bits, bits)
    print("Bit error rate after LDPC:", errorPostLDPC)

    
