import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ImageCaptioner import ImageCaptioner
import load

import settings

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

    model = model.to(settings.DEVICE)

    # define cross entropy loss
    loss_fn = nn.CrossEntropyLoss()

    for i in tqdm(range(num_epochs)):

        avg_loss = 0
        nbatches = 0

        for j, data in enumerate(dataloader):
            images, _, gt_toks, _, _ = data
            images = images.to(settings.DEVICE)
            gt_toks = gt_toks.to(settings.DEVICE)
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
        torch.save(model.state_dict(), settings.CHECKPOINT_PATH)

    return

def eval(model, dl):
    print("Evaluating")

    model = model.to(settings.DEVICE)

    for data in dl:
        imgs, _, gt_toks, _, _ = data
        imgs = imgs.to(settings.DEVICE)
        gt_toks = gt_toks.to(settings.DEVICE)

        result = model.sample_strings(imgs)
        print(result)
        import pdb; pdb.set_trace()

    return


def main(args):
    vocab = load.load_vocab(split="train")

    checkpoint = settings.CHECKPOINT_PATH if args.eval else None
    captioner = load.load_captioner(vocab, checkpoint=checkpoint) 

    total_params = sum(p.numel() for p in captioner.parameters())
    trainable_params = sum(p.numel() for p in captioner.parameters() if p.requires_grad)
    print(f"Vocab Size: {len(vocab)}")
    print(f"Total Parameters:{total_params / (1000000):.2f}M")
    print(f"Train Parameters:{trainable_params / (1000):.2f}K")

    if not args.eval:
        dl = load.load_dataloader(vocab, split="train", shuffle=True, batch_size=6)

        opt = torch.optim.Adam(captioner.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        train(captioner, dl, opt, scheduler)
    else:
        dl = load.load_dataloader(vocab, split="val", shuffle=False, batch_size=1)
        eval(captioner, dl)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()
    main(args)
