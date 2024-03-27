from eval.intrinsic.bert import NextSentenceMetric
from eval.intrinsic.clip import CLIPSimilarityMetric

import settings
import load


def evaluate_ns_metric(gen_fn, dl):
    '''gen_fn is a function that will take in a batch of images and produce
    a list of strings representing captions for each image in the batch
    
    images will be come from dl
    '''

    pass

def evaluate_clip_metric(gen_fn, dl):
    '''gen_fn is a function that will take in a batch of images and produce
    a list of strings representing captions for each image in the batch
    
    images will come from dl
    '''
    pass

def main():

    # load vocab and dataloader
    vocab = load.load_vocab(split="train")    
    dl = load.load_dataloader(vocab, split="val")

    # load the model
    captioner = load.load_captioner(vocab, settings.MODEL_SAVE_PATH)

    pass

if __name__ == "__main__":
    main()

