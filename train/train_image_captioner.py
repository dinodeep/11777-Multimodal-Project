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

RUN_NAME = "full_train_captioner"
MODLE_SAVE_PATH = f"models/save/{RUN_NAME}"


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

    for i in tqdm(range(num_epochs)):

        avg_loss = 0
        nbatches = 0

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
            avg_loss += loss.item()
            # import pdb; pdb.set_trace()

            # backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            nbatches += 1

            # break

        avg_loss /= nbatches
        print(f"Avg Loss: {avg_loss:.3f}")
        print(model.sample_strings(images))

        # save the model
        torch.save(model, MODLE_SAVE_PATH)

    return

def eval(model, dl):
    print("Evaluating")

    for data in dl:
        imgs, _, gt_toks, _, _ = data

        result = model.sample_token_idxs(imgs)
        import pdb; pdb.set_trace()

    return


def main(args):

    data_dir = args.data_dir
    FOLDER = "val"

    # TODO: incapsulate vocab creation a bit better
    root = os.path.join(data_dir, "images", FOLDER)
    sis_path = os.path.join(data_dir, "sis", f"{FOLDER}.story-in-sequence.json")

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

    SHUFFLE = True
    dl = get_dataloader(root, sis_path, vocab, TRAIN_TRANSFORM, max_seq_len, SHUFFLE, False, 0)

    opt = torch.optim.Adam(captioner.parameters(), lr=0.005)
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