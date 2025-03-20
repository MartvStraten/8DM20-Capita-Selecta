import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torchsummary import summary
import neptune
import os
from dotenv import load_dotenv
import utils
import vae
import matplotlib.pyplot as plt
from PIL import Image

# to ensure reproducible training/validation split
random.seed(41)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Starting Neptune session
load_dotenv()
LOG = True

if LOG:
    run = neptune.init_run(
        description="Frist try VAE",
        name="VAE_1_C",
        project=os.getenv('NEPTUNE_PROJECT_VAE'),
        api_token=os.getenv("NEPTUNE_KEY"),
        mode="debug"
    )  




# directorys with data and to store training checkpoints and logs

DATA_DIR = Path.cwd()/ "DevelopmentData"
print(DATA_DIR)
print(DATA_DIR)
CHECKPOINTS_DIR = Path.cwd() / "vae_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "vae_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 200
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 10

# dimension of VAE latent space
Z_DIM = 256


params = {
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam"
}
if LOG:
    run["parameters"] = params


# function to reduce the
def lr_lambda(the_epoch):
    """Function for scheduling learning rate"""
    return (
        1.0
        if the_epoch < DECAY_LR_AFTER
        else 1 - float(the_epoch - DECAY_LR_AFTER) / (N_EPOCHS - DECAY_LR_AFTER)
    )


# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)
print("Amount of folders found: ",len(patients))
# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling

dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE, valid=True) # in my experiments the augmentations
# did not help, so I set valid=True to disable them
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE, valid=True)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser
loss_function = vae.vae_loss
vae_model = vae.VAE()
vae_model.to(device)
#print(summary(vae_model, (1,64,64), 32))
optimizer = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)

# add a learning rate scheduler based on the lr_lambda function
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


# training loop
for epoch in range(N_EPOCHS):
    print("EPOCH: ",epoch)
    current_train_loss = 0.0
    current_valid_loss = 0.0

    for x_real, y_real in tqdm(dataloader, position=0):
        # needed to zero gradients in each iterations
        optimizer.zero_grad()
        outputs, mu, logvar = vae_model(x_real.to(device))  # forward pass
        loss = loss_function(x_real.to(device), outputs.to(device), mu=mu, logvar=logvar)
        loss.backward()  # backpropagate loss
        current_train_loss += loss
        optimizer.step()  # update weights



    scheduler.step() # step the learning step scheduler
    


    # evaluate validation loss
    with torch.no_grad():
        vae_model.eval()
        # evaluate validation loss
        for inputs, labels in tqdm(valid_dataloader, position=0):
            outputs, mu, logvar = vae_model(inputs.to(device))  # forward pass
            loss = loss_function(inputs.to(device), outputs.to(device), mu=mu, logvar=logvar)
            current_valid_loss += loss

        # save examples of real/fake images
        if (epoch + 1) % DISPLAY_FREQ == 0:
            x_recon = outputs
            img_grid = make_grid(
                torch.cat((x_recon[:5].cpu(), x_real.detach()[:5].cpu())), nrow=5, padding=12, pad_value=-1
            )
            save_image(img_grid, "buffer.png")
            if LOG:
                run["images/realfake"].append(value=neptune.types.File.as_image((np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5).squeeze()), 
                                              description=f'EPOCH: {epoch}')

            # writer.add_image(
            #     "Real_fake", np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5, epoch + 1
            # )
        
            # TODO: sample noise 

            noise = vae.get_noise(10, 256, device)

            # TODO: generate 10 images and display
            
            image_samples = vae_model.generator(noise)

            img_grid = make_grid(
                torch.cat((image_samples[:5].cpu(), image_samples[5:].cpu())),
                nrow=5,
                padding=12,
                pad_value=-1,
            )
            save_image(img_grid, "buffer.png")
            if LOG:
                run["images/reconstructions"].append(value=neptune.types.File.as_image((np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5).squeeze()), 
                                                     description=f'EPOCH: {epoch}')
        if LOG:
            run["train/loss"].append(current_train_loss)
            run["valid/loss"].append(current_valid_loss)
        vae_model.train()
run.stop()
weights_dict = {k: v.cpu() for k, v in vae_model.state_dict().items()}
torch.save(
    weights_dict,
    CHECKPOINTS_DIR / "vae_model.pth",
)
