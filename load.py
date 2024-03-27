'''file for loading vocabulary and model based on variable definitions in
settings.py
'''

from models.ImageCaptioner import ImageCaptioner
import settings

from data.dataloader import \
    build_vocab, \
    build_character_vocab, \
    get_dataset, \
    get_dataloader, \
    TRAIN_TRANSFORM, \
    VAL_TRANSFORM

import torch
import os

def _split_valid(split):
    return split == "train" or split == "val" or split == "test"

def load_image_root(split="train"):
    assert(_split_valid(split))
    return os.path.join(settings.DATA_DIR, "images", split)
    return 

def load_sis_path(split="train"):
    assert(_split_valid(split))
    return os.path.join(settings.DATA_DIR, "sis", f"{split}.story-in-sequence.json")

def load_vocab(split="train"):
    assert(_split_valid(split))
    sis_path = load_sis_path(split)

    if settings.USE_WORD_VOCAB:
        vocab = build_vocab(sis_path, settings.WORD_VOCAB_COUNT_THRESHOLD)
    else:
        vocab = build_character_vocab(sis_path, settings.CHAR_VOCAB_COUNT_THRESHOLD)

    return vocab

def load_dataset():
    # TODO: 
    pass

def load_dataloader(vocab, split="train", shuffle=False, batch_size=1):
    assert(_split_valid(split))

    root = load_image_root(split)
    sis = load_sis_path(split)

    transform = TRAIN_TRANSFORM if split == "train" else VAL_TRANSFORM
    max_seq_len = settings.WORD_MAX_SEQ_LEN if settings.USE_WORD_VOCAB else settings.CHAR_MAX_SEQ_LEN
    dl = get_dataloader(root, sis, vocab, transform, max_seq_len, batch_size, shuffle, 0) 

    return dl

def load_captioner(vocab, checkpoint=None):

    captioner = ImageCaptioner(
        vocab,
        token_emb_dim=settings.TOKEN_EMB_DIM,
        hidden_dim=settings.HIDDEN_DIM
    )

    if checkpoint is not None:
        captioner.load_state_dict(torch.load(checkpoint, map_location=settings.DEVICE))

    return captioner


if __name__ == "__main__":
    
    # perform an example load
    vocab = load_vocab(split="train")
    dataloader = load_dataloader(vocab, split="val")
    pass