import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import u_net
import utils
import neptune
import os
import dotenv
from torchvision.utils import make_grid, save_image
import numpy as np

dotenv.load_dotenv()

# to ensure reproducible training/validation split
random.seed(42)

# find out if a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path("DevelopmentData")
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 100
LEARNING_RATE = 1e-4
TOLERANCE = 0.03  # for early stopping
DISPLAY_FREQ = 10

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser, and loss function
loss_function = utils.DiceBCELoss()
unet_model = u_net.UNet(num_classes=1).to(device)
optimizer = torch.optim.Adam(unet_model.parameters(), lr=LEARNING_RATE)
minimum_valid_loss = 10  # initial validation loss

minimum_valid_loss = 10  # initial validation loss


# Initialize Neptune experiment
run = neptune.init_run(
        description="First try UNET",
        name="UNET_1_C",
        project=os.getenv('NEPTUNE_PROJECT_UNET'),
        api_token=os.getenv("NEPTUNE_KEY"),
        #mode="debug"
)

# Track hyperparameters
run["parameters"] = {
    "lr": LEARNING_RATE,
    "bs": BATCH_SIZE,
    "epochs": N_EPOCHS,
    "input_sz": IMAGE_SIZE[0] * IMAGE_SIZE[1],
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


# training loop
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0

    for inputs, labels in tqdm(dataloader, position=0):
        # needed to zero gradients in each iterations
        optimizer.zero_grad()
        outputs = unet_model(inputs.to(device))  # forward pass
        loss = loss_function(outputs, labels.to(device).float())
        loss.backward()  # backpropagate loss
        current_train_loss += loss.item()
        optimizer.step()  # update weights

    # evaluate validation loss
    with torch.no_grad():
        unet_model.eval()
        for inputs, labels in tqdm(valid_dataloader, position=0):
            outputs = unet_model(inputs.to(device))  # forward pass
            loss = loss_function(outputs, labels.to(device).float())
            current_valid_loss += loss.item()

        if (epoch + 1) % DISPLAY_FREQ == 0:
            img_grid = make_grid(
                torch.cat((inputs[:5].cpu(), labels.cpu()[:5], outputs.cpu()[:5])), nrow=5, padding=12, pad_value=-1
            )
            run["image/predictions"].append(value=neptune.types.File.as_image((np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5).squeeze()), 
                                            description=f'EPOCH: {epoch}')

        unet_model.train()

    # Write to neptune log
    run["train/loss"].append(current_train_loss)
    run["valid/loss"].append(current_valid_loss)

    # if validation loss is improving, save model checkpoint
    # only start saving after 10 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        weights_dict = {k: v.cpu() for k, v in unet_model.state_dict().items()}
        if epoch > 9:
            torch.save(
                weights_dict,
                CHECKPOINTS_DIR / f"u_net.pth",
            )

# End neptune run
run.stop()