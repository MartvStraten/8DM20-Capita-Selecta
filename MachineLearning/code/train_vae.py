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
from optuna.trial import TrialState
import vae
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
import neptune.integrations.optuna as npt_utils
import sklearn

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


run = neptune.init_run(
    description="Parameter Sweep",
    name="Parameter Sweep",
    project=os.getenv('NEPTUNE_PROJECT_VAE'),
    api_token=os.getenv("NEPTUNE_KEY"),
    #mode="debug"
)  

neptune_callback = npt_utils.NeptuneCallback(run)



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
N_EPOCHS = 50
DECAY_LR_AFTER = 50
LEARNING_RATE = 1e-4
DISPLAY_FREQ = 10

# dimension of VAE latent space
#Z_DIM = 256



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
def load_set():
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
    return dataloader, valid_dataloader

def objective(trial: optuna.trial):
    dataloader, valid_dataloader = load_set()
    # initialise model, optimiser
    loss_function = vae.vae_loss
    zdim = trial.suggest_categorical("z_dim", [256, 2*256, 4*256])
    reluvalue = trial.suggest_float("Relu_value", 0.01, 0.5)
    betavalue = trial.suggest_float("Beta_value", 0.01, 1.4)
    vae_model = vae.VAE(z_dim=zdim, relu_value=reluvalue).to(device)
    #print(summary(vae_model, (1,64,64), 32))
    optimizer_name = "Adam"

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = getattr(optim, optimizer_name)(vae_model.parameters(), lr=lr)

    # add a learning rate scheduler based on the lr_lambda function
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    for epoch in range(N_EPOCHS):
        print("EPOCH: ",epoch)
        current_train_loss = 0.0
        current_valid_loss = 0.0

        for x_real, y_real in tqdm(dataloader, position=0):
            # needed to zero gradients in each iterations
            optimizer.zero_grad()
            outputs, mu, logvar = vae_model(x_real.to(device))  # forward pass
            loss = loss_function(x_real.to(device), outputs.to(device), mu=mu, logvar=logvar, beta=betavalue)
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
                loss = loss_function(inputs.to(device), outputs.to(device), mu=mu, logvar=logvar, beta=betavalue)
                current_valid_loss += loss

            # save examples of real/fake images
            if (epoch + 1) % DISPLAY_FREQ == 0:
                x_recon = outputs
                img_grid = make_grid(
                    torch.cat((x_recon[:5].cpu(), x_real.detach()[:5].cpu())), nrow=5, padding=12, pad_value=-1
                )
            
                run["images/"+str(trial.number)+"/realfake"].append(value=neptune.types.File.as_image((np.clip(img_grid[0][np.newaxis], -1, 1) / 2 + 0.5).squeeze()), 
                                            description=f'EPOCH: {epoch}')
            
            run["train/"+str(trial.number)+"/loss"].append(current_train_loss)
            run["valid/"+str(trial.number)+"/loss"].append(current_valid_loss)
            kld_loss = vae.kld_loss(mu.cpu(),logvar.cpu())
            run["valid/"+str(trial.number)+"/KLDloss"].append(kld_loss)
            trial.report(kld_loss, epoch) 
            vae_model.train()

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if np.isnan(kld_loss):
                kld_loss = 0
    return kld_loss


if __name__ == "__main__":
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=100,
            callbacks=[neptune_callback]
        )
    except KeyboardInterrupt:
        print("Stopping study")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    run["no_finished_trials"].append(len(study.trials))
    run["no_pruned_trials"].append(len(pruned_trials))
    run["no_complete_rials"].append(len(complete_trials))

    print("Best trial:")
    trial = study.best_trial


    print("  Value: ", trial.value)
    run.stop()
