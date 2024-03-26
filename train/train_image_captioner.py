import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ImageCaptioner import ImageCaptioner
from data.dataloader import \
    build_vocab, \
    build_character_vocab, \
    get_dataset, \
    get_dataloader, \
    TRAIN_TRANSFORM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, opt, num_epochs=100):
    '''
        going to define training process

        given a dataloader that loads data
        given an optimizer over the parameters
        given number of epochs

        define a loss function
    '''
    print("Training")

    model = model.to(DEVICE)

    # define cross entropy loss
    loss_fn = nn.CrossEntropyLoss()

    for i in range(num_epochs):

        for j, data in enumerate(dataloader):
            print(j)
            images, _, gt_toks, _, _ = data
            images = images.to(DEVICE)
            gt_toks = gt_toks.to(DEVICE)
            # print(gt_toks)

            # import pdb; pdb.set_trace()

            # forward pass
            pred = model(images, gt_toks)
            # print(pred.argmax(-1))

            # get the next sentence gt tokens (appropriately ignore last model output)
            next_gt_toks = gt_toks[:, 1:]
            pred = pred[:, :-1, :]

            # re-shape output to treat each prediction equaly
            pred = pred.flatten(end_dim=-2)
            next_gt_toks = next_gt_toks.flatten()

            loss = loss_fn(pred, next_gt_toks)
            print(loss.item())
            # import pdb; pdb.set_trace()

            # backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()


            # break
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

    use_word_tokenizer = True

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
        token_emb_dim=1024, # 512,
        hidden_dim=2048 # 1024
    )
    # captioner.train()

    total_params = sum(p.numel() for p in captioner.parameters())
    trainable_params = sum(p.numel() for p in captioner.parameters() if p.requires_grad)
    print(f"Vocab Size: {len(vocab)}")
    print(f"Total Parameters:{total_params / (1000000):.2f}M")
    print(f"Train Parameters:{trainable_params / (1000):.2f}K")

    # TODO: should be moving this to data loader with appropriate collating
    dl = get_dataloader(root, sis_path, vocab, TRAIN_TRANSFORM, max_seq_len, 6, False, 0)

    opt = torch.optim.Adam(captioner.parameters(), lr=0.001)
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