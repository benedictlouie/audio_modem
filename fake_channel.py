from utils import *

def generate_channel():
    length = 128
    decay_factor = 0.95
    impulse_response = np.random.randn(length) * (decay_factor ** np.arange(length))
    return impulse_response

if __name__ == "__main__":
    signal = load_audio_file(AUDIO_PATH)
    channel = generate_channel()
    signal = np.convolve(signal, channel, mode="full")
    signal += np.random.normal(0, 0.1, signal.shape)
    write_wav(AUDIO_PATH, signal)


