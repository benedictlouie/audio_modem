import numpy as np
import sounddevice as sd
import sys
import threading

from utils.utils import write_wav, get_original_bits, get_symbols_from_bitstream, decode_bits_to_file, open_file_with_default_app
from utils.parameters import *
from utils.plot import plot_sent_received_constellation, plot_received_constellation
from decoder import iterative_decoder

def record_until_enter(samplerate=48000, channels=1):
    print("Recording... Press Enter to stop.")
    recording = []
    stop_event = threading.Event()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        recording.append(indata.copy())

    def wait_for_enter():
        input()
        stop_event.set()

    enter_thread = threading.Thread(target=wait_for_enter)
    enter_thread.start()

    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        while not stop_event.is_set():
            sd.sleep(100)

    audio = np.concatenate(recording, axis=0)
    print("Recording stopped.")
    return audio

if __name__ == "__main__":
    signal = record_until_enter().flatten()
    sd.stop()
    write_wav(RECEIVED_AUDIO_PATH, signal)

    received_data, received_symbols = iterative_decoder(signal)

    if KNOWN_RECEIVER:
        original_bits = get_original_bits()
        sent_symbols = get_symbols_from_bitstream(original_bits)
        plot_sent_received_constellation(sent_symbols, received_symbols)
        received_data = received_data[:len(original_bits)]
        print(f'Bit Error Rate after LDPC: {np.sum(received_data != original_bits) / len(original_bits) * 100:.2f}%')
    else:
        plot_received_constellation(received_symbols)

    path = decode_bits_to_file(received_data)
    open_file_with_default_app(path)