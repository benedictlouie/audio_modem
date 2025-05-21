from utils import *

def generate_channel():
    length = 3000
    decay_factor = 0.95
    impulse_response = np.random.randn(length) * (decay_factor ** np.arange(length))
    return impulse_response

if __name__ == "__main__":
    signal = load_audio_file(AUDIO_PATH)
    channel = generate_channel()
    signal = np.convolve(signal, channel, mode="full")
    signal += 0.1 * np.random.normal(0, 0.1, signal.shape)
    write_wav("noisy.wav", signal)


