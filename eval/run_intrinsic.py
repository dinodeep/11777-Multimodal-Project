from eval.intrinsic.bert import NextSentenceMetric
from eval.intrinsic.clip import CLIPSimilarityMetric

import settings
import load


def evaluate_ns_metric_gen(gen_fn, dl):
    '''gen_fn is a function that will take in a batch of images and produce
    a list of strings representing captions for each image in the batch
    
    images will be come from dl
    '''

    # initialize metric
    ns_metric = NextSentenceMetric()
    probs = []

    for batch in dl:
        imgs, gt_strs, gt_toks, _, _ = batch

        prob = ns_metric.next_sentence_predict_seq(gen_fn(imgs))
        probs.append(prob)

        print(sum(probs) / len(probs))
        

    return sum(probs) / len(probs)

def evaluate_clip_metric_gen(gen_fn, dl):
    '''gen_fn is a function that will take in a batch of images and produce
    a list of strings representing captions for each image in the batch
    
    images will come from dl
    '''

    # initialize metric
    clip_metric = CLIPSimilarityMetric()
    scores = []

    for batch in dl:
        imgs, gt_strs, gt_toks, _, _ = batch

        captions = gen_fn(imgs)
        score = clip_metric.similarity_seq(imgs, captions)
        scores.append(score)

        print(sum(scores) / len(scores))
        

    return sum(scores) / len(scores)

    pass


def captioner_to_gen_fn(captioner):

    def gen_fn(images):
        '''images is (B, C, H, W) -> list[str] (list is of length B)'''
        captions =captioner.sample_strings(images) 
        return captions

    return gen_fn 

def evaluate(gen_fn, dl):

    evaluate_ns_metric_gen(gen_fn, dl)
    evaluate_clip_metric_gen(gen_fn, dl)

    return

def main():

    # load vocab and dataloader
    vocab = load.load_vocab(split="train")    
    dl = load.load_dataloader(vocab, split="val")

    # load the model
    captioner = load.load_captioner(vocab, settings.CHECKPOINT_PATH)
    gen_fn = captioner_to_gen_fn(captioner)

    evaluate(gen_fn, dl)
    return

if __name__ == "__main__":
    main()

