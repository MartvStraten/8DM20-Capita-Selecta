import numpy as np
import random
import torch
import torchio as tio
import SimpleITK as sitk

from torch.utils.data import DataLoader
from pathlib import Path

import utils
import cvae


def bin_masks(dataloader, bin_edges):
    # initialise 16 bins
    binned_masks =[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for _, total_segment in dataloader:
        # Loop over individual segmentations
        for segment in total_segment:
            # Calculate number of pixels
            segment_np = segment.squeeze().numpy()
            num_pixels = np.sum(segment_np)
            bin_id = min(np.digitize(num_pixels, bin_edges) - 1, 15)
            binned_masks[bin_id].append(segment.to(torch.int32).unsqueeze(0))
            
    return binned_masks


def floodfill_data(bins, bin_edges, binned_masks, cvae_model):
    for folder_nr in range(1,16):
        print(f"populating folder: cvae ({folder_nr})")
        new_mr = []
        new_masks = []
        path = f"D:\TUe\8DM20\8DM20-Capita-Selecta\DevelopmentData\cvae ({folder_nr})"
        for _ in range(86):
            # find bin with least entries
            lowest_bin = np.argmin(bins)
            # pick random mask from lowest bin
            mask = random.choice(binned_masks[lowest_bin])

            # apply reldef untill mask is not too large
            reldef = tio.RandomElasticDeformation(max_displacement=(0,15,15), locked_borders=2)
            aug_mask = torch.ceil(reldef(mask))
            num_pixels = int(np.sum(aug_mask.squeeze().numpy()))
            while num_pixels > 1000:
                reldef = tio.RandomElasticDeformation(max_displacement=(0,15,15), locked_borders=2)
                aug_mask = torch.ceil(reldef(mask))
                num_pixels = int(np.sum(aug_mask.squeeze().numpy()))
            
            # generate image with cvae
            noise = torch.randn(1, Z_DIM)
            generated_mr = cvae_model.generator(noise, aug_mask)

            # add mask and generated image to arrays
            new_mr.append(generated_mr)
            new_masks.append(aug_mask)

            bin_id = min(np.digitize(num_pixels, bin_edges) - 1, 15)
            bins[bin_id] += 1

        new_mr = torch.cat(new_mr, 0).detach().numpy()
        new_masks = torch.cat(new_masks, 0).detach().numpy()
        sitk.WriteImage(sitk.GetImageFromArray(new_mr), path+"\mr_bffe.mhd")
        sitk.WriteImage(sitk.GetImageFromArray(new_masks), path+"\prostaat.mhd")


if __name__ == "__main__":
    # directorys with data and to store training checkpoints and logs
    DATA_DIR = Path.cwd() / "DevelopmentData"
    CHECKPOINTS_DIR = r"D:\TUe\8DM20\8DM20-Capita-Selecta\MachineLearning\code\cvae_model_weights\cvae_model_new_loss.pth"
    print(DATA_DIR)
    print(CHECKPOINTS_DIR)

    # data settings 
    IMAGE_SIZE = [64, 64]
    BATCH_SIZE = 1290
    Z_DIM = 256

    # find patient folders in training directory
    # excluding hidden folders (start with .)
    patients = [
        path
        for path in DATA_DIR.glob("*")
        if not any(part.startswith(".") or part.startswith("c") for part in path.parts)
    ]
    print(patients)
    # load training data and create DataLoader with batching and shuffling
    dataset = utils.ProstateMRDataset(patients, IMAGE_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    # Obtain all images in a tensor
    total_prostate, total_segment = next(iter(dataloader))

    # Load the trained model weights
    cvae_model = cvae.CVAE()
    cvae_model.load_state_dict(torch.load(CHECKPOINTS_DIR))
    cvae_model.eval();

    pixel_count = []
    # Loop over segmentation inside dataloader
    for total_prostate, total_segment in dataloader:
        # Loop over individual segmentations
        for segment in total_segment:
            # Calculate number of pixels
            segment = segment.squeeze().numpy()
            num_pixels = np.sum(segment)
            pixel_count.append(int(num_pixels))
    pc_array = np.array(pixel_count)

    pc_bins, pc_bin_edges = np.histogram(pc_array, 16)
    binned_masks = bin_masks(dataloader, pc_bin_edges)

    floodfill_data(pc_bins, pc_bin_edges, binned_masks, cvae_model);
    