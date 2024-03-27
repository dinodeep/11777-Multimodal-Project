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

RUN_NAME = "large_vocab_train_captioner"
MODEL_SAVE_PATH = f"models/save/{RUN_NAME}"

MAX_BATCHES_PER_EPOCH = 100


def train(model, dataloader, opt, scheduler, num_epochs=100):
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
            images, _, gt_toks, _, _ = data
            images = images.to(DEVICE)
            gt_toks = gt_toks.to(DEVICE)
            # print(gt_toks[0])

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
            print(j, loss)

            # backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            nbatches += 1

            if j >= MAX_BATCHES_PER_EPOCH:
                break

        scheduler.step()
        
        avg_loss /= nbatches
        print(f"Avg Loss: {avg_loss:.3f}")
        print(model.sample_strings(images[:1, ...]))

        # save the model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    return

def eval(model, dl):
    print("Evaluating")

    model = model.to(DEVICE)

    for data in dl:
        imgs, _, gt_toks, _, _ = data
        imgs = imgs.to(DEVICE)
        gt_toks = gt_toks.to(DEVICE)

        result = model.sample_strings(imgs)
        print(result)
        import pdb; pdb.set_trace()

    return


def main(args):

    data_dir = args.data_dir
    FOLDER = "train"

    # TODO: incapsulate vocab creation a bit better
    root = os.path.join(data_dir, "images", FOLDER)
    sis_path = os.path.join(data_dir, "sis", f"{FOLDER}.story-in-sequence.json")

    use_word_tokenizer = True

    MAX_CHAR_SEQ_LEN = 64
    MAX_WORD_SEQ_LEN = 32
    
    if use_word_tokenizer:
        max_seq_len = MAX_WORD_SEQ_LEN
        vocab = build_vocab(sis_path, 10) # originally 50
    else:
        max_seq_len = MAX_CHAR_SEQ_LEN
        vocab = build_character_vocab(sis_path, 50)


    captioner = ImageCaptioner(
        vocab=vocab,
        token_emb_dim=2048, # 1024,
        hidden_dim=4096 # 2048
    )
    
    if args.eval:
        captioner.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print("loading model for evaluation successful")

    total_params = sum(p.numel() for p in captioner.parameters())
    trainable_params = sum(p.numel() for p in captioner.parameters() if p.requires_grad)
    print(f"Vocab Size: {len(vocab)}")
    print(f"Total Parameters:{total_params / (1000000):.2f}M")
    print(f"Train Parameters:{trainable_params / (1000):.2f}K")

    if not args.eval:
        SHUFFLE = True
        BATCH_SIZE = 6
        dl = get_dataloader(root, sis_path, vocab, TRAIN_TRANSFORM, max_seq_len, BATCH_SIZE, SHUFFLE, 0)

        opt = torch.optim.Adam(captioner.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        train(captioner, dl, opt, scheduler)
    else:
        SHUFFLE = False
        BATCH_SIZE = 1
        dl = get_dataloader(root, sis_path, vocab, TRAIN_TRANSFORM, max_seq_len, BATCH_SIZE, SHUFFLE, 0)
        eval(captioner, dl)


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
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()
    main(args)
