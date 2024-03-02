'''A lot of code adapted from GLAC competitive baseline model'''

import nltk
import pickle
import argparse
from collections import Counter
import json

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
import re
from PIL import Image

IMG_SIZE = 224

TRAIN_TRANSFORM = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        ])

VAL_TRANSFORM = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225))
        ])

# download nltk setup
nltk.download("wordnet")

class VIST:
    def __init__(self, sis_file = None):
        if sis_file != None:
            sis_dataset = json.load(open(sis_file, 'r'))
            self.LoadAnnotations(sis_dataset)


    def LoadAnnotations(self, sis_dataset = None):
        images = {}
        stories = {}

        if 'images' in sis_dataset:
            for image in sis_dataset['images']:
                images[image['id']] = image

        if 'annotations' in sis_dataset:
            annotations = sis_dataset['annotations']
            for annotation in annotations:
                story_id = annotation[0]['story_id']
                stories[story_id] = stories.get(story_id, []) + [annotation[0]]

        self.images = images
        self.stories = stories

class Vocabulary(object):
    def __init__(self, tokenizer, merger):
        '''tokenizer: str -> list of str constructing tokens'''
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.tokenizer = tokenizer
        self.merger = merger
        
        self.PAD = '<pad>'
        self.START = '<start>'
        self.END = '<end>'
        self.UNK = '<unk>'

        self._add_default_tokens()

    def _add_default_tokens(self):
        self.add_word(self.PAD)
        self.add_word(self.START)
        self.add_word(self.END)
        self.add_word(self.UNK)
        return
    
    def get_padding_idx(self):
        return self.word2idx[self.PAD]

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.UNK]
        return self.word2idx[word]
    
    def i2w(self, i):
        if i not in self.idx2word:
            return self.idx2word[self.UNK]
        return self.idx2word[i]

    def __len__(self):
        return len(self.word2idx)
    

def build_vocab(sis_file, threshold):
    vist = VIST(sis_file, )
    counter = Counter()

    def tokenizer(s):
        return nltk.tokenize.word_tokenize(s.lower())
    
    def merger(tokens):
        return " ".join(tokens)

    ids = vist.stories.keys()
    for i, id in enumerate(ids):
        story = vist.stories[id]
        for annotation in story:
            caption = annotation['text']
            tokens = []
            try:
                tokens = tokenizer(caption)
            except Exception:
                pass
            counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the story captions." %(i, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary(tokenizer, merger)

    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab

def build_character_vocab(sis_file, threshold):

    vist = VIST(sis_file)
    counter = Counter()

    def tokenizer(s):
        clean_text = re.sub(r'[^a-zA-Z0-9_ \.?!:]+', '', s.lower())
        return list(clean_text)
    
    def merger(tokens):
        return "".join(tokens)

    ids = vist.stories.keys()
    for i, id in enumerate(ids):
        story = vist.stories[id]
        for annotation in story:
            caption = annotation["text"]

            for c in caption:
                counter.update(c)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the story captions." %(i, len(ids)))

    chars = [char for char, cnt in counter.items() if cnt >= threshold] 

    vocab = Vocabulary(tokenizer, merger)

    for i, chars in enumerate(chars):
        vocab.add_word(chars)

    return vocab

class VistDataset(data.Dataset):
    def __init__(self, image_dir, sis_path, vocab, max_seq_len=64, transform=None):
        self.image_dir = image_dir
        self.vist = VIST(sis_path)
        self.ids = list(self.vist.stories.keys())
        self.vocab = vocab
        self.transform = transform
        self.max_seq_len = max_seq_len


    def __getitem__(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = []
        raw_targets = []
        images = []
        photo_sequence = []
        album_ids = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, str(image_id) + image_format)).convert('RGB')
                except Exception:
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            tokens = []
            # print("ANNOTATION", text.lower())
            # try:
            tokens = self.vocab.tokenizer(text)
            # except Exception :
                # print("tokenizing caption failed")
                # pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            caption = caption[:self.max_seq_len]

            if len(caption) < self.max_seq_len:
                caption += [vocab('<pad>')] * (self.max_seq_len - len(caption))

            target = torch.Tensor(caption)
            targets.append(target)
            raw_targets.append(text.lower())

        images = torch.stack(images)
        targets = torch.stack(targets).to(torch.int64)
        return images, raw_targets, targets, photo_sequence, album_ids


    def __len__(self):
        return len(self.ids)


def collate_fn(data):

    # NOTE: because each sample starts with 5 samples
    # resulting number of samples is 5 * batch_size for image captioning
    # TODO: separate data loading functionality: image captioning vs. story gen

    imgs = [data[i][0] for i in range(len(data))]
    raw_targets = [data[i][1] for i in range(len(data))]
    targets = [data[i][2] for i in range(len(data))]
    photo_sequence = [data[i][3] for i in range(len(data))]
    album_ids = [data[i][4] for i in range(len(data))]

    # convert the images and targets into a single tensor
    imgs = torch.concat(imgs, dim=0)
    targets = torch.concat(targets, dim=0)

    return imgs, raw_targets, targets, photo_sequence, album_ids

def get_dataset(root, sis_path, vocab, transform, max_seq_len):
    vist = VistDataset(image_dir=root, sis_path=sis_path, vocab=vocab, transform=transform, max_seq_len=max_seq_len)
    return vist

def get_dataloader(root, sis_path, vocab, transform, max_seq_len, batch_size, shuffle, num_workers):
    vist = VistDataset(image_dir=root, sis_path=sis_path, vocab=vocab, transform=transform, max_seq_len=max_seq_len)
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader

def test():

    ROOT = "data/raw-data/images/val"
    SIS_PATH = "data/raw-data/sis/val.story-in-sequence.json"
    use_word_tokenizer = False

    MAX_CHAR_SEQ_LEN = 64
    MAX_WORD_SEQ_LEN = 32
    
    if use_word_tokenizer:
        max_seq_len = MAX_WORD_SEQ_LEN
        vocab = build_vocab(SIS_PATH, 50)
    else:
        max_seq_len = MAX_CHAR_SEQ_LEN
        vocab = build_character_vocab(SIS_PATH, 50)

    ds = get_dataset(ROOT, SIS_PATH, vocab, TRAIN_TRANSFORM, max_seq_len)
    data = ds[0]

    '''
    description of data
        data[0]: tensor of stack of images (5, C, H, W) - normalized with mean 0 and std. dev something
        data[1]: list[str] - list of length 5, each item is ground truth text, untokenized
        data[2]: tensor of shape (5, max_seq_len), each a tensor of some varying length representing indices into vocabulary (padded if necessary)
        data[3]: list[str] - list of string image ids
        data[4]: list[str] - list of different album ids or something?
    '''

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test()