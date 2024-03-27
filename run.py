import train.train_image_captioner as train

import argparse

def main(args):
    train.main(args)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args()
    main(args)
