import utils.ldpc as ldpc

# Frequently-used Settings
AUDIO_PATH = "output.wav"               # Output audio after encoding
RECEIVED_AUDIO_PATH = AUDIO_PATH
RECEIVED_AUDIO_PATH = "received.wav"    # Input audio for decoding
FILE_PATH = "received_5.tiff"
KNOWN_RECEIVER = False
LDPC_RATE = '5/6'
SHIFT_BACK = 3
RESIDUAL_THRESHOLD = 1 # For the RANSAC model

SAMPLE_RATE = 48000

# Number of symbols [1:N_DFT//2] per block
EFFECTIVE_SYMBOLS_PER_BLOCK = 2 ** 12 - 1
CYCLIC_PREFIX = 2 ** 11
BLOCK_LENGTH = 2 * (EFFECTIVE_SYMBOLS_PER_BLOCK + 1) + CYCLIC_PREFIX
N_DFT = BLOCK_LENGTH - CYCLIC_PREFIX

# Cut-offs because of hardware limitation
LOW_PASS_INDEX = 2143
HIGH_PASS_INDEX = 199

# Number of symbols after cut-offs
SYMBOLS_PER_BLOCK = LOW_PASS_INDEX - HIGH_PASS_INDEX

# Number of information blocks following each known block
INFORMATION_BLOCKS_PER_FRAME = 4
FRAME_LENGTH = (INFORMATION_BLOCKS_PER_FRAME + 1) * BLOCK_LENGTH

# Chirp
CHIRP_TIME = 0.5
CHIRP_LENGTH = round(CHIRP_TIME * SAMPLE_RATE)
CHIRP_FACTOR = 0.04
CHIRP_LOW = 0
CHIRP_HIGH = 3000

# QPSK
BITS_PER_SYMBOL = 2
assert BITS_PER_SYMBOL == 2

# LDPC Settings
DECTYPE = 'sumprod2'
CODE = ldpc.code(z=81, rate=LDPC_RATE)