

def show(imgs):
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()

def plot_normalized_images(images):
    '''images are (B, C, H, W)'''

    from torchvision.utils import make_grid
    import torch

    # B, C, H, W
    images = images * torch.tensor([0.229, 0.224, 0.225]).reshape((1, -1, 1, 1)) + torch.tensor([0.485, 0.456, 0.406]).reshape((1, -1, 1, 1))
    grid = make_grid(images)

    show(grid)

    return