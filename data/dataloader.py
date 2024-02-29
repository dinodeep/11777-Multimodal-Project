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
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
    

def build_vocab(sis_file, threshold):
    vist = VIST(sis_file, )
    counter = Counter()

    ids = vist.stories.keys()
    for i, id in enumerate(ids):
        story = vist.stories[id]
        for annotation in story:
            caption = annotation['text']
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(caption.lower())
            except Exception:
                pass
            counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized the story captions." %(i, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab

class VistDataset(data.Dataset):
    def __init__(self, image_dir, sis_path, vocab, transform=None):
        self.image_dir = image_dir
        self.vist = VIST(sis_path)
        self.ids = list(self.vist.stories.keys())
        self.vocab = vocab
        self.transform = transform


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
        print("Start of get item")
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
            print("ANNOTATION", text.lower())
            # try:
            tokens = nltk.tokenize.word_tokenize(text.lower())
            # except Exception :
                # print("tokenizing caption failed")
                # pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)
            raw_targets.append(text.lower())
        print("here")
        return torch.stack(images), raw_targets, targets, photo_sequence, album_ids


    def __len__(self):
        return len(self.ids)


def collate_fn(data):

    image_stories, caption_stories, photo_sequence_set, album_ids_set = zip(*data)

    targets_set = []
    lengths_set = []

    for captions in caption_stories:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        targets_set.append(targets)
        lengths_set.append(lengths)

    return image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set


def get_dataset(root, sis_path, vocab, transform):
    vist = VistDataset(image_dir=root, sis_path=sis_path, vocab=vocab, transform=transform)
    return vist

def get_loader(root, sis_path, vocab, transform, batch_size, shuffle, num_workers):
    vist = VistDataset(image_dir=root, sis_path=sis_path, vocab=vocab, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader