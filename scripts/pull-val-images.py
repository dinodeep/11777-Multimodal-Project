import load
import settings
import utils.vision as vision


def main():

    vocab = load.load_vocab(split="train")
    val_ds = load.load_dataset(vocab, split="val")

    model = load.load_captioner(vocab, settings.CHECKPOINT_PATH)

    for idx in [2493, 2706, 2725, 2134, 5]:
         imgs, raw_targets, targets, photo_sequence, album_ids = val_ds[idx]

         strs = model.sample_strings(imgs)
         print(" ".join(strs))
         print("=======")
         vision.plot_normalized_images(imgs)


    return

if __name__ == "__main__":
    main()