import load
import utils.vision as vision

def main():

    vocab = load.load_vocab(split="train")
    val_ds = load.load_dataset(vocab, split="val")

    for idx in [2493, 2706, 2725, 2134, 5]:
         imgs, raw_targets, targets, photo_sequence, album_ids = val_ds[idx]
         vision.plot_normalized_images(imgs)

         import pdb; pdb.set_trace()


    pass

if __name__ == "__main__":
    main()