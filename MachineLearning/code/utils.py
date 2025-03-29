import random

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size, valid=False):
        self.mr_image_list = []
        self.mask_list = []
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd")).astype(
                    np.int32
                )
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(
                    np.int32
                )
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        if valid:
            self.img_transform = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    transforms.CenterCrop(256),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            )
        else:
            # transforms to resize images
            self.img_transform = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    transforms.CenterCrop(256),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.RandomAffine(
                        degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                ]
            )
        # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.

        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """
        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)
        torch.manual_seed(seed)

        x = self.norm_transform(
            self.img_transform(self.mr_image_list[patient][the_slice, ...]).float()
        )
        random.seed(seed)
        torch.manual_seed(seed)
        y = self.img_transform(
            (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
        )
        return x, y


class DiceBCELoss(nn.Module):
    """Loss function, computed as the sum of Dice score and binary cross-entropy.

    Notes
    -----
    This loss assumes that the inputs are logits (i.e., the outputs of a linear layer),
    and that the targets are integer values that represent the correct class labels.
    """

    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, outputs, targets, smooth=1):
        """Calculates segmentation loss for training

        Parameters
        ----------
        outputs : torch.Tensor
            predictions of segmentation model
        targets : torch.Tensor
            ground-truth labels
        smooth : float
            smooth parameter for dice score avoids division by zero, by default 1

        Returns
        -------
        float
            the sum of the dice loss and binary cross-entropy
        """
        outputs = torch.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # compute Dice
        intersection = (outputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            outputs.sum() + targets.sum() + smooth
        )
        BCE = nn.functional.binary_cross_entropy(outputs, targets, reduction="mean")

        return BCE + dice_loss


class DatasetCVAE(torch.utils.data.Dataset):   
    def __init__(self, paths, valid=False):
        self.mr_image_list = []
        self.mask_list = []
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd"))[:,0,:,:].astype(
                    np.int32
                )
            )
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd"))[:,0,:,:].astype(
                    np.int32
                )
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        if valid:
            self.img_transform = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    transforms.ToTensor(),
                ]
            )
        else:
            # transforms to resize images
            self.img_transform = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    transforms.ToTensor(),
                    transforms.RandomAffine(
                        degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                ]
            )
        # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.

        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """
        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)
        torch.manual_seed(seed)

        x = self.norm_transform(
            self.img_transform(self.mr_image_list[patient][the_slice, ...]).float()
        )
        random.seed(seed)
        torch.manual_seed(seed)
        y = self.img_transform(
            (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
        )
        return x, y


class CombinedProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images.

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """

    def __init__(self, paths, img_size, valid=False):
        self.mr_image_list = []
        self.mask_list = []
        # load images
        for path in paths:
            mr_image = sitk.GetArrayFromImage(sitk.ReadImage(path / "mr_bffe.mhd")).astype(np.int32)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(path / "prostaat.mhd")).astype(np.int32)
            if len(mr_image.shape) == 4:
                mr_image = mr_image.squeeze(1)
                mask = mask.squeeze(1)
            self.mr_image_list.append(mr_image)
            self.mask_list.append(mask)

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        if valid:
            self.img_transform = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    transforms.CenterCrop(256),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ]
            )
            self.img_transform_cvae = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.img_transform = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    transforms.CenterCrop(256),
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.RandomAffine(
                        degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                ]
            )
            self.img_transform_cvae = transforms.Compose(
                [
                    transforms.ToPILImage(mode="I"),
                    transforms.ToTensor(),
                    transforms.RandomAffine(
                        degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                ]
            )
        
        total_sum = 0
        total_sq_sum = 0
        total_count = 0
        for arr in self.mr_image_list:
            total_sum += arr.sum()
            total_sq_sum += (arr**2).sum()  # Sum of squares
            total_count += arr.size

        mean_value = total_sum / total_count
        std_value = np.sqrt((total_sq_sum / total_count) - (mean_value ** 2))
        self.train_data_mean = mean_value
        self.train_data_std = std_value

        # standardise intensities based on mean and std deviation
        # self.train_data_mean = np.mean(self.mr_image_list)
        # self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset"""
        return self.no_patients * self.no_slices

    def __getitem__(self, index):
        """Returns the preprocessing MR image and corresponding segementation
        for a given index.

        Parameters
        ----------
        index : int
            index of the image/segmentation in dataset
        """
        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)
        torch.manual_seed(seed)
        x = self.mr_image_list[patient][the_slice, ...]
        if x.shape[-1] == 64:
            x = self.norm_transform(self.img_transform_cvae(self.mr_image_list[patient][the_slice, ...]).float())
        else:
            x = self.norm_transform(self.img_transform(self.mr_image_list[patient][the_slice, ...]).float())
        
        random.seed(seed)
        torch.manual_seed(seed)
        y = self.mask_list[patient][the_slice, ...] > 0
        if y.shape[-1] == 64:
            y = self.img_transform_cvae((y).astype(np.int32))
        else:
            y = self.img_transform((y).astype(np.int32))
        
        return x, y
