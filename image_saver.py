import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.axis("off")

def save_images(images: torch.tensor, idx: int=0, num_images: int = 25):
    images = (images + 1) / 2
    image_unflat = images.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    ax.imshow(image_grid.permute(1, 2, 0).squeeze())
    fig.savefig(f"./output/gen_{idx}.png", dpi=200)


def save_showcase(gen, dataloader, getNoise, device, epochs=[1, 10, 80]):
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("WGAN-GP Results - Trained on Celeba-50k")

    for i in range(4):
        row = i//2
        col = i%2
        ax[row][col].axis("off")
        if i == 0:
            for real in dataloader:
                ax[row][col].set_title("Real Images")
                images = real
                break
        else:
            gen.load_state_dict(torch.load(f"checkpoints/gen_state_{epochs[i-1]}"))
            ax[row][col].set_title(f"Epoch {epochs[i-1]}")
            noise = getNoise(25, device=device)
            images = gen(noise)
        images = (images + 1) / 2
        image_unflat = images.detach().cpu()
        image_grid = make_grid(image_unflat[:25], nrow=5)
        ax[row][col].imshow(image_grid.permute(1, 2, 0).squeeze())

    fig.savefig("./showcase.png", dpi=200)