from eval.intrinsic.bert import NextSentenceMetric
from eval.intrinsic.clip import CLIPSimilarityMetric

import settings
import load

MAX_EVAL_BATCHES = 100

def show(imgs):
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()

def make_grid_batch(images):
    from torchvision.utils import make_grid
    import torch

    # B, C, H, W
    images = images * torch.tensor([0.229, 0.224, 0.225]).reshape((1, -1, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).reshape((1, -1, 1, 1))
    grid = make_grid(images)

    show(grid)

    return


def captioner_to_gen_fn(captioner):

    def gen_fn(images):
        '''images is (B, C, H, W) -> list[str] (list is of length B)'''
        captions =captioner.sample_strings(images) 
        return captions

    return gen_fn 

def evaluate_gen_fn(gen_fn, dl):

     # initialize metric
    ns_metric = NextSentenceMetric()
    probs = []

    # initialize metric
    clip_metric = CLIPSimilarityMetric()
    scores = []

    for i, batch in enumerate(dl):
        if i >= MAX_EVAL_BATCHES:
            break

        imgs, gt_strs, gt_toks, _, _ = batch

        captions = gen_fn(imgs)
        prob = ns_metric.next_sentence_predict_seq(captions)
        score = clip_metric.similarity_seq(imgs, captions)

        probs.append(prob)
        scores.append(score)

        for caption in captions:
            print("\t", caption)
        print(i, sum(probs) / len(probs), sum(scores) / len(scores))
        
    return sum(probs) / len(probs), sum(scores) / len(scores)

def main():

    # load vocab and dataloader
    vocab = load.load_vocab(split="train")    
    dl = load.load_dataloader(vocab, split="val", shuffle=True)

    # load the model
    captioner = load.load_captioner(vocab, settings.CHECKPOINT_PATH)
    gen_fn = captioner_to_gen_fn(captioner)

    evaluate_gen_fn(gen_fn, dl)
    return

if __name__ == "__main__":
    main()

