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

def load_dataloader(vocab, split="train", shuffle=False, batch_size=1):
    '''vocab's split and split should be the same'''
    assert(_split_valid(split))

    root = load_image_root(split)
    sis = load_sis_path(split)

    transform = TRAIN_TRANSFORM if split == "train" else VAL_TRANSFORM
    dl = get_dataloader(root, sis, vocab, transform, settings.MAX_SEQ_LEN, batch_size, shuffle, 0) 

    return dl


if __name__ == "__main__":
    
    # perform an example load
    pass