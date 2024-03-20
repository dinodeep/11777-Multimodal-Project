import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn

from models.ImageCaptioner import ImageCaptioner
from data.dataloader import \
    build_vocab, \
    build_character_vocab, \
    get_dataset, \
    get_dataloader, \
    TRAIN_TRANSFORM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, opt, num_epochs=10):
    '''
        going to define training process

        given a dataloader that loads data
        given an optimizer over the parameters
        given number of epochs

        define a loss function
    '''
    print("Training")

    LIMIT = 1

    model = model.to(DEVICE)

    # define cross entropy loss
    loss_fn = nn.CrossEntropyLoss()

    for i in range(num_epochs):

        for data in tqdm(dataloader):
            images, _, gt_toks, _, _ = data
            images = images.to(DEVICE)
            gt_toks = gt_toks.to(DEVICE)
            print(gt_toks)

            # forward pass
            pred = model(images, gt_toks)

            # re-shape output to treat each prediction equaly
            pred = pred.flatten(end_dim=-2)
            gt_toks = gt_toks.flatten()
            loss = loss_fn(pred, gt_toks)
            print(loss.item())

            # backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(model.sample_strings(images))

    return

def eval(model, dl):
    print("Evaluating")

    for data in dl:
        imgs, _, gt_toks, _, _ = data

        result = model.sample_token_idxs(imgs)
        import pdb; pdb.set_trace()


    pass


def main(args):

    data_dir = args.data_dir

    # TODO: incapsulate vocab creation a bit better
    root = os.path.join(data_dir, "images", "val")
    sis_path = os.path.join(data_dir, "sis", "val.story-in-sequence.json")

    use_word_tokenizer = False

    MAX_CHAR_SEQ_LEN = 64
    MAX_WORD_SEQ_LEN = 32
    
    if use_word_tokenizer:
        max_seq_len = MAX_WORD_SEQ_LEN
        vocab = build_vocab(sis_path, 50)
    else:
        max_seq_len = MAX_CHAR_SEQ_LEN
        vocab = build_character_vocab(sis_path, 50)


    captioner = ImageCaptioner(
        vocab=vocab,
        token_emb_dim=512,
        hidden_dim=512
    )

    # TODO: should be moving this to data loader with appropriate collating
    dl = get_dataloader(root, sis_path, vocab, TRAIN_TRANSFORM, max_seq_len, 1, False, 0)

    opt = torch.optim.Adam(captioner.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        5,
        0.1
    )

    train(captioner, dl, opt)


if __name__ == "__main__":

    # data folder should be structured as follows
    '''
        {data-dir}/
            images/
                train/
                val/
                test/
            dii/
            sis/
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")

    args = parser.parse_args()
    main(args)