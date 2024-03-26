import torch
import torch.nn as nn

import torchvision

class ImageEncoder(nn.Module):

    def __init__(self, img_emb_dim=256):
        super().__init__()

        # dimension of the image encoding
        self.img_emb_dim = img_emb_dim 

        # initialize frozen resnetv151 model with classifier head removed
        resnet_dim = 2048
        weights = torchvision.models.ResNet152_Weights.DEFAULT
        resnet = torchvision.models.resnet152(weights)
        resnet_modules = list(resnet.children())[:-1]

        # create image encoder with custom image encoding dimension
        self.img_enc = nn.Sequential(
            *resnet_modules,
            nn.Flatten(),
            nn.Linear(2048, self.img_emb_dim)
        )

        return

    def forward(self, x):
        '''x: (B, C, H, W)'''
        z = self.img_enc(x)
        return z 
    


def test():
    enc = ImageEncoder()
    total_params = sum(p.numel() for p in enc.parameters())
    print(f"Total Parameters: {total_params / 1000000}")

    b = 8
    c = 3
    d = 224
    x = torch.randn((b, c, 224, 224))
    e = enc(x)
    print("Forward pass of input (b, 3, 244, 244) successful")
    return

if __name__ == "__main__":
    test() 