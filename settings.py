'''file containing configuration parameters for training, data loading, and vocabs'''

# model settings
MODEL = "large_vocab_train_captioner"
TOKEN_EMB_DIM = 2048
HIDDEN_DIM = 4096

# vocab settings
USE_WORD_VOCAB = True
WORD_VOCAB_COUNT_THRESHOLD = 10 # originally 50
CHAR_VOCAB_COUNT_THRESHOLD = 50
MAX_SEQ_LEN = 50


# data loading
DATA_DIR = "data/raw-data"