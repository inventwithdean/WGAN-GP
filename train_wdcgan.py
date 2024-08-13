import torch
from torch import nn
from tqdm.auto import tqdm

from generator import *
from critic import *
from dataset_loader import getDataLoader
from image_saver import save_images, save_showcase

LEARNING_RATE = 0.0002
BETA_1 = 0
BETA_2 = 0.9
BATCH_SIZE = 128
DEVICE = "cuda"
C_LAMBDA = 10
CRIT_UPDATES = 5
N_EPOCHS = 100

dataloader = getDataLoader(dir="./50k", batch_size=BATCH_SIZE)
gen = Generator(im_channels=3).to(DEVICE)
crit = Critic(im_channels=3).to(DEVICE)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, mean=0, std=0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=0, std=0.02)
        nn.init.constant_(m.bias, 0)


gen.apply(weights_init)
crit.apply(weights_init)

# last_trained_epoch = 80
# gen.load_state_dict(torch.load(f"checkpoints/gen_state_{last_trained_epoch}"))
# crit.load_state_dict(torch.load(f"checkpoints/disc_state_{last_trained_epoch}"))

gen_opt = torch.optim.Adam(gen.parameters(), LEARNING_RATE, betas=(BETA_1, BETA_2))
crit_opt = torch.optim.Adam(crit.parameters(), LEARNING_RATE, betas=(BETA_1, BETA_2))

for epoch in range(1, N_EPOCHS+1):
    for real in tqdm(dataloader):
        real = real.to(DEVICE)
        current_batch_size = len(real)

        for _ in range(CRIT_UPDATES):
            # HANDLING CRITIC
            crit_opt.zero_grad(set_to_none=True)
            crit_real_pred = crit(real)

            noise = getNoise(current_batch_size, device=DEVICE)        
            generated_images = gen(noise).detach()
            crit_fake_pred = crit(generated_images)

            epsilon = torch.randn(len(real), 1, 1, 1, device=DEVICE, requires_grad=True)
            gradient = get_gradient(crit, real, generated_images, epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, C_LAMBDA)
            crit_loss.backward(retain_graph=True)

            crit_opt.step()

        # HANDLING GENERATOR
        gen_opt.zero_grad(set_to_none=True)
        noise = getNoise(current_batch_size, device=DEVICE)
        generated_images = gen(noise)
        crit_pred = crit(generated_images)
        gen_loss = get_gen_loss(crit_pred)
        gen_loss.backward()
        gen_opt.step()
    print(f"Epoch: {epoch}: Generator Loss: {gen_loss.item():2f} | Critic Loss: {crit_loss.item():2f}")
    torch.save(gen.state_dict(), f"./checkpoints/gen_state_{epoch}")
    torch.save(crit.state_dict(), f"./checkpoints/disc_state_{epoch}")
    save_images(generated_images, epoch)

save_showcase(gen, dataloader, getNoise, DEVICE, epochs=[1, 10, 100])
