from torch.utils import data
from os import listdir
import os
import imageio
import numpy as np
import torch
import h5py
from PIL import Image

class SintefDatasetValTest(data.Dataset):
    """Class to load the Sintef MRI US Dataset.
    """

    def __init__(self, rootDir, mriFilePrefix, usFilePrefix):
        """
        Args:
            rootDir (string): Directory with all the images. 
        """
        self.rootDir = rootDir

        self.mriFilePrefix = mriFilePrefix
        self.usFilePrefix = usFilePrefix
        self.imageSize = (64, 64)

        self.usFiles = {}
        self.mriFiles = {}

        self.datasetSize = 0
        for root, dirs, files in os.walk(self.rootDir):
            for file in files:
                if file.endswith('.h5') and file.startswith(usFilePrefix):
                    usFileTemp = os.path.join(self.rootDir, file)
                    if os.path.isfile(usFileTemp):
                        usVolumeFile = h5py.File(usFileTemp, 'r')
                        usVolumeFileKey = list(usVolumeFile.keys())[0]
                        usVolume = usVolumeFile[usVolumeFileKey]
                        

                        mriFile = usFileTemp.replace(self.usFilePrefix, self.mriFilePrefix)
                        mriFile = mriFile.replace('.h5', '_0_0_0.imf.h5')
                        mriVolumeFile = h5py.File(mriFile, 'r')
                        mriVolumeFileKey = list(mriVolumeFile.keys())[0]
                        mriVolume = mriVolumeFile[mriVolumeFileKey]
                        mriVolume.shape[0]
                        #quick sanity check
                        if(mriVolume.shape[0] == usVolume.shape[0]):
                            self.datasetSize += usVolume.shape[0]
                            self.usFiles[usFileTemp] = usVolume.shape[0]
                        else:
                            print ('Ignoring this data file because they had different number of slices:\n', mriFileTemp, usFileTemp)
                
        print("Dataset Details:")
        print(self.usFiles)
        print("Total dataset size: ", self.datasetSize)

    def __len__(self):
        return self.datasetSize

    #dummy function
    def __getitem__(self, idx):
        usFile = []
        frameCount = 0
        for key, value in sorted(self.usFiles.items()):
            frameCount += value
            if idx < frameCount:
                usFile = key;
                break

        usVolumeFile = h5py.File(usFile, 'r')
        usVolumeFileKey = list(usVolumeFile.keys())[0]
        usVolume = usVolumeFile[usVolumeFileKey]
        usVolumeSize = usVolume.shape[0]
        usImage = usVolume[-frameCount + usVolumeSize + idx, 0, :, :]

        usImage = Image.fromarray(usImage.astype(float)/255.0)
        usImage = usImage.resize(self.imageSize, Image.ANTIALIAS)
        usImage = np.array(usImage)

        usImage = usImage[:, :, np.newaxis]
        usImage = usImage.transpose((2, 0, 1))
        usImage = torch.tensor(usImage, dtype=torch.float, requires_grad=False)

        mriFile = usFile.replace(self.usFilePrefix, self.mriFilePrefix)
        mriFile = mriFile.replace('.h5', '_0_0_0.imf.h5')

        
        rotValues = [-20, -10, -5, 0, 5, 10, 20 ]
        x = [-5, -3, -1, 0, 1, 3, 5 ]
        y = [-5, -3, -1, 0, 1, 3, 5 ]

        rotatedMris = torch.Tensor([])

        for rot in rotValues:

            #for tx in x:
                #for ty in y:
                    tx= 0
                    ty = 0
                    mriFile = mriFile.split('_')[0] + '_' + str(tx) + '_' + str(ty) + '_' + str(rot) + '.imf.h5'
                    mriVolumeFile = h5py.File(mriFile, 'r')
                    mriVolumeFileKey = list(mriVolumeFile.keys())[0]
                    mriVolume = mriVolumeFile[mriVolumeFileKey]
                    mriVolumeSize = mriVolume.shape[0]
                    mriImage = mriVolume[-frameCount + mriVolumeSize + idx, 0, :, :]

                    mriImage = Image.fromarray(mriImage.astype(float)/1500.0)
                    mriImage = mriImage.resize(self.imageSize, Image.ANTIALIAS)
                    mriImage = np.array(mriImage)    
                    mriImage = mriImage[:, :, np.newaxis]
                    mriImage = mriImage.transpose((2, 0, 1))
                    mriImage = torch.tensor(mriImage, dtype=torch.float, requires_grad=False)

                    #True mri pair
                    if rot == 0 & tx == 0 & ty == 0:                    
                        trueMri = mriImage

                    #rotated mri's
                    else:
                        rotatedMris = torch.cat((rotatedMris, mriImage), 0)

        sample = {'mri': trueMri, 'us': usImage, 'rotatedMris': rotatedMris}

        return sample
        

    def __getus__(self, idx):
        usFile = []
        frameCount = 0
        for key, value in sorted(self.usFiles.items()):
            frameCount += value
            if idx < frameCount:
                usFile = key;
                break

        usVolumeFile = h5py.File(usFile, 'r')
        usVolumeFileKey = list(usVolumeFile.keys())[0]
        usVolume = usVolumeFile[usVolumeFileKey]
        usVolumeSize = usVolume.shape[0]
        usImage = usVolume[-frameCount + usVolumeSize + idx, 0, :, :]

        usImage = Image.fromarray(usImage.astype(float)/255.0)
        usImage = usImage.resize(self.imageSize, Image.ANTIALIAS)
        usImage = np.array(usImage)

        usImage = usImage[:, :, np.newaxis]
        usImage = usImage.transpose((2, 0, 1))
        usImage = torch.tensor(usImage, dtype=torch.float, requires_grad=False)

        return usImage

    def __getmri__(self, idx, tx, ty, rot):

        usFile = []
        frameCount = 0
        for key, value in sorted(self.usFiles.items()):
            frameCount += value
            if idx < frameCount:
                usFile = key;
                break

        mriFile = usFile.replace(self.usFilePrefix, self.mriFilePrefix)
        mriFile = mriFile.replace('.h5', '_' + str(tx) + '_' + str(ty) + '_' + str(rot) + '.imf.h5')

        mriVolumeFile = h5py.File(mriFile, 'r')
        mriVolumeFileKey = list(mriVolumeFile.keys())[0]
        mriVolume = mriVolumeFile[mriVolumeFileKey]
        mriVolumeSize = mriVolume.shape[0]
        mriImage = mriVolume[-frameCount + mriVolumeSize + idx, 0, :, :]

        mriImage = Image.fromarray(mriImage.astype(float)/1500.0)
        mriImage = mriImage.resize(self.imageSize, Image.ANTIALIAS)
        mriImage = np.array(mriImage)

        mriImage = mriImage[:, :, np.newaxis]
        mriImage = mriImage.transpose((2, 0, 1))
        mriImage = torch.tensor(mriImage, dtype=torch.float, requires_grad=False)

        return mriImage
