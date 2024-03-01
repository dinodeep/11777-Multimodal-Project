import torch
import torch.nn as nn
import torchvision

from models.ImageEncoder import ImageEncoder
from data.dataloader import build_vocab, build_character_vocab, get_dataset, TRAIN_TRANSFORM

class ImageCaptioner(nn.Module):

    def __init__(self, vocab, token_emb_dim=256, hidden_dim=512):
        super().__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)

        # initialize image encoder
        self.hidden_dim = hidden_dim
        self.img_encoder = ImageEncoder(hidden_dim)

        # initialize token embeddings
        self.token_emb_dim = token_emb_dim
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.token_emb_dim,
            padding_idx=self.vocab.get_padding_idx()
        )

        # initialize RNN layer to create RNN outputs
        self.rnn = nn.RNN(
            input_size=self.token_emb_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        # create final layer map to vocab dimension
        self.lin_out = nn.Linear(self.hidden_dim, self.vocab_size)

        # softmax to create probability distribution over vocabulary
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, imgs, gt_words):
        # x (B, C, H, W)
        # tokens: (B, L)

        assert (imgs.dim() == 4 and gt_words.dim() == 2)
        assert (imgs.shape[0] == gt_words.shape[0])

        B, C, H, W = imgs.shape
        _, L = gt_words.shape

        # image features: (B, D)
        img_enc = self.img_encoder(imgs)

        # encoded tokens: (B, L, E)
        tok_enc = self.embeddings(gt_words)

        # pass into img_enc as initial hidden state and tokens for prediction
        img_enc = img_enc.unsqueeze(0)
        rnn_out, h_n = self.rnn(tok_enc, img_enc)

        # transform tokens into (B, L, V) using linear transform into vocab size
        out = self.lin_out(rnn_out)

        # softmax to get outputs
        prob = self.softmax(out)

        # (B, L, V)
        B_out, L_out, V_out = prob.shape
        assert(B_out == B and L_out == L and V_out == self.vocab_size)
        return prob
 

def test():

    ROOT = "data/raw-data/images/val"
    SIS_PATH = "data/raw-data/sis/val.story-in-sequence.json"
    use_word_tokenizer = False

    MAX_CHAR_SEQ_LEN = 64
    MAX_WORD_SEQ_LEN = 32
    
    if use_word_tokenizer:
        max_seq_len = MAX_WORD_SEQ_LEN
        vocab = build_vocab(SIS_PATH, 50)
    else:
        max_seq_len = MAX_CHAR_SEQ_LEN
        vocab = build_character_vocab(SIS_PATH, 50)


    ds = get_dataset(ROOT, SIS_PATH, vocab, TRAIN_TRANSFORM, max_seq_len)
    imgs, _, gt_toks, _, _= ds[0]

    # import pdb; pdb.set_trace()

    captioner = ImageCaptioner(
        vocab,
        token_emb_dim=16,
        hidden_dim=16
    )

    out = captioner(imgs, gt_toks)

    return


if __name__ == "__main__":
    test()




