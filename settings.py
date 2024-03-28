'''file containing configuration parameters for training, data loading, and vocabs'''

import torch

# model settings
MODEL = "large_vocab_train_captioner"
TOKEN_EMB_DIM = 2048
HIDDEN_DIM = 4096

CHECKPOINT_PATH = f"models/save/{MODEL}" # derived - NO TOUCH

# data loading
DATA_DIR = "data/raw-data"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# vocab settings
USE_WORD_VOCAB = True

WORD_VOCAB_COUNT_THRESHOLD = 10 # originally 50
WORD_MAX_SEQ_LEN = 32

CHAR_VOCAB_COUNT_THRESHOLD = 50
CHAR_MAX_SEQ_LEN = 64

VOCAB_PICKLE_DIR = f"{DATA_DIR}/vocab"
