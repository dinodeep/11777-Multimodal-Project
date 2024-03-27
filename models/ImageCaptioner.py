import torch
import torch.nn as nn
import torchvision

from models.ImageEncoder import ImageEncoder
from data.dataloader import build_vocab, build_character_vocab, get_dataset, TRAIN_TRANSFORM
import settings

class ImageCaptioner(nn.Module):

    def __init__(self, vocab, token_emb_dim=256, hidden_dim=512):
        super().__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)

        # initialize image encoder
        self.hidden_dim = hidden_dim
        self.img_encoder = ImageEncoder(hidden_dim)
        self.img_encoder.eval()
        self._freeze_img_enc()

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

    def _freeze_img_enc(self):
        for p in self.img_encoder.parameters():
            p.requires_grad = False

    def train(self):
        super().train()
        self._freeze_img_enc() 
            
        return

    def eval(self):
        super().eval()
        self._freeze_img_enc()

        return


    def forward(self, imgs, gt_words):
        '''takes in images and predicts words by feeding in ground truth words'''
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
        # prob = self.softmax(out)

        # (B, L, V)
        B_out, L_out, V_out = out.shape
        assert(B_out == B and L_out == L and V_out == self.vocab_size)
        return out
    

    def sample_token_idxs(self, imgs, max_length=32):
        '''takes in a batch of images and generates sequences of text'''
        # imgs: (B, C, H, W)
        B, _, _, _ = imgs.shape

        # encode images: (B, D)
        img_enc = self.img_encoder(imgs)

        # create start tokens for each sequence in the batch (B, 1, E)
        start_indices = torch.full((B, 1), self.vocab.get_starting_idx()).to(settings.DEVICE)
        start_tok_embeddings = self.embeddings(start_indices).to(settings.DEVICE)

        # initialize token embeddings being passed into RNN layer and hidden
        seq_token_idxs = [start_indices]
        curr_token_embeddings = start_tok_embeddings
        h = img_enc.unsqueeze(0)

        # iterate through and keep adding tokens until done
        for i in range(max_length):
            # get hidden unit and mkae predictions on sequence
            z, h = self.rnn(curr_token_embeddings, h)
            next_token_logits = self.lin_out(z)
            next_token_probs = self.softmax(next_token_logits)
            next_token_idxs = torch.argmax(next_token_probs, dim=-1)

            # decide on the next tokens and get their embeddings to recurse
            seq_token_idxs.append(next_token_idxs)
            curr_token_embeddings = self.embeddings(next_token_idxs)


        # convert the tokens into a tensor
        seq_idxs = torch.concat(seq_token_idxs, dim=-1)
        return seq_idxs
 
    def sample_tokens(self, imgs, max_length=32):

        # get the tokens
        tokens_idxs = self.sample_token_idxs(imgs, max_length=max_length)
        batch_token_idxs = tokens_idxs.cpu().tolist()

        # convert each of them into their appropriate strings
        batch_token_list = []
        for token_idxs in batch_token_idxs:
            token_list = []
            for idx in token_idxs:
                token_list.append(self.vocab.i2w(idx))
            batch_token_list.append(token_list)

        return batch_token_list

    def sample_strings(self, imgs, max_length=32):

        batch_token_list = self.sample_tokens(imgs, max_length=max_length)

        # join each string token by a space
        batch_strs = []
        for token_list in batch_token_list:
            batch_strs.append(self.vocab.merger(token_list))
        
        return batch_strs

 

def test():

    ROOT = "data/raw-data/images/val"
    SIS_PATH = "data/raw-data/sis/val.story-in-sequence.json"
    use_word_tokenizer = True 

    MAX_CHAR_SEQ_LEN = 64
    MAX_WORD_SEQ_LEN = 32
    
    if use_word_tokenizer:
        max_seq_len = MAX_WORD_SEQ_LEN
        vocab = build_vocab(SIS_PATH, 50)
    else:
        max_seq_len = MAX_CHAR_SEQ_LEN
        vocab = build_character_vocab(SIS_PATH, 50)


    batch_size = 8


    ds = get_dataloader(
        ROOT, SIS_PATH, vocab, TRAIN_TRANSFORM, 
        max_seq_len, batch_size, True, 0
    )
    imgs, _, gt_toks, _, _= ds[0]

    # import pdb; pdb.set_trace()

    captioner = ImageCaptioner(
        vocab,
        token_emb_dim=16,
        hidden_dim=16
    )

    out = captioner(imgs, gt_toks)

    result = captioner.sample_strings(imgs)

    import pdb; pdb.set_trace()

    return


if __name__ == "__main__":
    test()




