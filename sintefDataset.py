from torch.utils import data
from os import listdir
import os
import imageio
import numpy as np
import torch
from PIL import Image
import h5py
from time import sleep

class SintefDataset(data.Dataset):
    """Class to load the Sintef MRI US Dataset.
    """

    def __init__(self, rootDir):
        """
        Args:
            rootDir (string): Directory with all the images. 
        """
        self.rootDir = rootDir

        self.mriFilePrefix =  'T1-DS-'
        self.usFilePrefix = 'US-DS-'
        self.imageSize = (64, 64)

        self.usFiles = {}
        self.mriFiles = {}
        self.datasetSize = 0

        idx = 0
        for root, dirs, files in os.walk(self.rootDir):
            for file in files:
                if file.endswith('.h5') and file.startswith(self.usFilePrefix):
                    usFileTemp = os.path.join(self.rootDir, file)
                    mriFileTemp = usFileTemp.replace(self.usFilePrefix, self.mriFilePrefix)
                    if os.path.isfile(mriFileTemp) and os.path.isfile(usFileTemp):
                        mriVolumeFile = h5py.File(mriFileTemp, 'r')
                        mriVolumeFileKey = list(mriVolumeFile.keys())[0]
                        mriVolume = mriVolumeFile[mriVolumeFileKey]

                        usVolumeFile = h5py.File(usFileTemp, 'r')
                        usVolumeFileKey = list(usVolumeFile.keys())[0]
                        usVolume = usVolumeFile[usVolumeFileKey]

                        if usVolume.shape[0] == mriVolume.shape[0]:
                            for localIdx in range(0, usVolume.shape[0]):
                                self.usFiles[idx] = (usFileTemp, localIdx)
                                self.mriFiles[idx] = (mriFileTemp, localIdx)
                                idx += 1
                        else:
                            print ('Ignoring this data file because they had different number of slices:\n', mriFileTemp, usFileTemp)

        self.datasetSize = idx
        print("Total dataset size: ", self.datasetSize)

    def __len__(self):
        return self.datasetSize

    def __getitem__(self, idx):
        (mriFile, localIdx) = self.mriFiles[idx]
        mriVolumeFile = h5py.File(mriFile, 'r')
        mriVolumeFileKey = list(mriVolumeFile.keys())[0]
        mriVolume = mriVolumeFile[mriVolumeFileKey]
        mriImage = mriVolume[localIdx, 0, :, :]

        mriImage = Image.fromarray(mriImage.astype(float)/1500.0)
        mriImage = mriImage.resize(self.imageSize, Image.ANTIALIAS)
        mriImage = np.array(mriImage)
        
        mriImage = mriImage[:, :, np.newaxis]
        mriImage = mriImage.transpose((2, 0, 1))
        mriImage = torch.tensor(mriImage, dtype=torch.float, requires_grad=False)

        (usFile, localIdx) = self.usFiles[idx]
        usVolumeFile = h5py.File(usFile, 'r')
        usVolumeFileKey = list(usVolumeFile.keys())[0]
        usVolume = usVolumeFile[usVolumeFileKey]
        usImage = usVolume[localIdx, 0, :, :]

        usImage = Image.fromarray(usImage.astype(float)/255.0)
        usImage = usImage.resize(self.imageSize, Image.ANTIALIAS)
        usImage = np.array(usImage)

        usImage = usImage[:, :, np.newaxis]
        usImage = usImage.transpose((2, 0, 1))
        usImage = torch.tensor(usImage, dtype=torch.float, requires_grad=False)

        sample = {'mri': mriImage, 'us': usImage}

        return sample