from torch.utils import data
from os import listdir
import os
import imageio
import numpy as np
import torch
class MriUsDataset(data.Dataset):
    """Class to load the MRI US Dataset.

    Currently has the limitation that 'us' and 'mri' folders must exist
    in the root directory. Also the subdirectories must have the images and its mask,
    named as 1.png for the us/mri image and 1m.png for the corresponding us/mri mask
    """

    def __init__(self, rootDir):
        """
        Args:
            rootDir (string): Directory with all the images. Must have two sub-directories: us and mri. Which has the US and MRI images and masks.
        """
        self.rootDir = rootDir

        self.datasetLength = len([f for f in os.listdir(os.path.join(self.rootDir, 'mri')) if f.endswith('.png') and os.path.isfile(os.path.join(self.rootDir, 'mri', f))])
        self.datasetLength = int(self.datasetLength/2)
    def __len__(self):
        return self.datasetLength

    def __getitem__(self, idx):
        mriImagePath = os.path.join(self.rootDir, 'mri', str(idx) + '.png')
        mriImage = imageio.imread(mriImagePath)
        mriImage = mriImage[:, :, np.newaxis]
        mriImage = mriImage.transpose((2, 0, 1))
        mriImage = torch.tensor(mriImage, dtype=torch.float, requires_grad=False)

        mriMaskPath = os.path.join(self.rootDir, 'mri', str(idx) + 'm.png')
        mriMask = imageio.imread(mriMaskPath)
        mriMask = mriMask[:, :, np.newaxis]
        mriMask = mriMask.transpose((2, 0, 1))
        mriMask = torch.tensor(mriMask, dtype=torch.float, requires_grad=False)

        usImagePath = os.path.join(self.rootDir, 'us', str(idx) + '.png')
        usImage = imageio.imread(usImagePath)
        usImage = usImage[:, :, np.newaxis]
        usImage = usImage.transpose((2, 0, 1))
        usImage = torch.tensor(usImage, dtype=torch.float, requires_grad=False)

        usMaskPath = os.path.join(self.rootDir, 'us', str(idx) + 'm.png')
        usMask = imageio.imread(usMaskPath)
        usMask = usMask[:, :, np.newaxis]
        usMask = usMask.transpose((2, 0, 1))
        usMask = torch.tensor(usMask, dtype=torch.float, requires_grad=False)

        sample = {'mri': mriImage, 'mriMask': mriMask, 'us': usImage, 'usMask': usMask}

        return sample
