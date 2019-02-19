import sys
import os
import sintefDataset as dataset
import sintefDatasetValTest as TestDataset
import matplotlib.pyplot as plt
import torch as torch
import torch.utils.data as batchloader
import unet_model as models
from unet_parts import *
from perceptualModel import AutoEncoder
import visdomLogger as vl
import numpy as np
import gc


# Constants
dx = torch.cuda.FloatTensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]]).detach()

dy = torch.cuda.FloatTensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]).detach()
dx = dx.view((1,1,3,3))
dy = dy.view((1,1,3,3))

def gradient(image):
    G_x = torch.nn.functional.conv2d(image, dx)
    G_y = torch.nn.functional.conv2d(image, dy)

    return torch.sqrt(torch.pow(G_x, 2)+ torch.pow(G_y, 2))

def calculateTotalLoss(mriOut, usOut, mriOutEncoded, usOutEncoded, mriEncoded, usEncoded, l2Loss, l1Loss):
    # calculate losses
    zeroTensor = torch.cuda.FloatTensor(mriOut.size()).fill_(0).detach()
    mriUsOutDiffLoss = l1Loss(mriOut - usOut, zeroTensor)

    zeroTensor2 = torch.cuda.FloatTensor(mriOutEncoded.size()).fill_(0).detach()
    mriEnLoss = l1Loss(mriOutEncoded - mriEncoded, zeroTensor2)
    usEnLoss = l1Loss(usOutEncoded - usEncoded, zeroTensor2)
    
    #enforce sparsity in the encoded vectors
    #sparsityLoss = l1Loss(mriOutEncoded, zeroTensor2) + l1Loss(usOutEncoded, zeroTensor2)
    totalLoss = 10000*mriUsOutDiffLoss + 450*mriEnLoss + 200*usEnLoss# + 50*sparsityLoss
    return (totalLoss, 10000*mriUsOutDiffLoss, 450*mriEnLoss, 200*usEnLoss, torch.Tensor([0]))

def main(mriEncoderPath, usEncoderPath):

    numEpochs, batchSize = 50, 16

    # Use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # create visdom logger instance
    logger = vl.VisdomLogger(port="65531")
    logger.deleteWindow(win="test")

    # Initialize the MRI US dataset and the dataset loader
    dir = os.path.dirname(__file__)
    modelPath = os.path.join(dir, '..', 'savedModels/transformation')

    trainDatasetDir = os.path.join(dir, '..', 'dataset/sintefHdf5/train')
    trainDataset = dataset.SintefDataset(rootDir=trainDatasetDir)

    valDatasetDir = os.path.join(dir, '..', 'dataset/sintefHdf5/val')
    validationDataset = dataset.SintefDataset(rootDir=valDatasetDir)

    dataloader = batchloader.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=0)
    
    
    # Initialize the MRI US dataset
    dir = os.path.dirname(__file__)
    testDatasetDir = '/media/alok/Data/deepLearning/datasets/sintefWolfgangAffine/tranRotMriVol/train'
    testDataset = TestDataset.SintefDatasetValTest(rootDir=testDatasetDir, mriFilePrefix='T1-Vol', usFilePrefix='US-DS')
    
    #create transformation models and add them to optimizer
    mriForwardmodel  = models.UNet(n_classes=1, n_channels=1).to(device)
    usForwardmodel   = models.UNet(n_classes=1, n_channels=1).to(device)
    usShallowModel = models.ShallowNet(n_classes=1, n_channels=1).to(device)
    mriShallowModel = models.ShallowNet(n_classes=1, n_channels=1).to(device)
    networkParams = list(mriForwardmodel.parameters()) + list(usForwardmodel.parameters()) + list(mriShallowModel.parameters()) + list(usShallowModel.parameters())
    optimizer = torch.optim.Adam(networkParams, lr=1e-5, weight_decay=1e-9)

    #Optimizer learning rate decay.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.98)

    #Load encoder models
    mriEncoderModel = AutoEncoder().to(device)
    usEncoderModel  = AutoEncoder().to(device)
    mriEncoderModel.load_state_dict(torch.load(mriEncoderPath))
    usEncoderModel.load_state_dict(torch.load(usEncoderPath))
    mriEncoderModel.eval()
    usEncoderModel.eval()

    # create instance of loss class
    l2Loss = torch.nn.MSELoss()
    l1Loss = torch.nn.L1Loss()

    iteration = 0
    for epochNum in range(50):
        for i_batch, sample_batched in enumerate(dataloader):
            mri = sample_batched['mri'].to(device)
            us = sample_batched['us'].to(device)

            # perform optimization
            optimizer.zero_grad()

            # Compute the shared representation
            mriOut = mriForwardmodel(mri)
            usOut = usForwardmodel(us)
            #StdMean Normalization, later Sigmoid assures the range of the image is between 0-1
            #mriOut = meanStdNormalize(mriOut, targetMean=0.0, targetStd=1)
            #usOut = meanStdNormalize(usOut, targetMean=0.0, targetStd=1)
            #mriOut = rangeNormalize(mriOut)
            #usOut = rangeNormalize(usOut)

            # Compute the encoded Representations
            (_, _, _, _, mriOutEncoded) = mriEncoderModel(mriShallowModel(mriOut), get_features = True)
            (_, _, _, _, usOutEncoded) = usEncoderModel(usShallowModel(usOut), get_features = True)
            (_, _, _, _, mriEncoded) = mriEncoderModel(mri, get_features = True)
            (_, _, _, _, usEncoded) = usEncoderModel(us, get_features = True)

             # Calculate Losses
            (totalLoss, mriUsOutDiffLoss, mriEnLoss, usEnLoss, sparsityLoss) = calculateTotalLoss(mriOut, usOut, mriOutEncoded, usOutEncoded, mriEncoded, usEncoded, l2Loss, l1Loss)

            # perform backward pass
            totalLoss.backward()

            optimizer.step()

            # print Epoch number and Loss
            print("Epoch: ", epochNum, "iteration: ", i_batch, "Total loss: ", totalLoss.item(),
                  "MriUSLoss: ", mriUsOutDiffLoss.item(), "MriEncoderLoss: ", mriEnLoss.item(), "UsEncoderLoss:", usEnLoss.item())

            iteration = iteration + 1
            logger.appendLine(X=np.array([iteration]), Y=np.array([totalLoss.item()]), win="test", name="Total Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([mriUsOutDiffLoss.item()]), win="test", name="MRI US Diff Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([mriEnLoss.item()]), win="test", name="MRI Encoder Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([usEnLoss.item()]), win="test", name="US Encoder Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([sparsityLoss.item()]), win="test", name="Sparsity Loss", xlabel="Iteration Number", ylabel="Loss")

            usValIn = torch.Tensor([])
            mriValIn = torch.Tensor([])

            if iteration%20 == 0:

                #switch model to eval mode for validation
                mriForwardmodel.eval()
                usForwardmodel.eval()

                # Show output on a validation image after some iterations
                valBatchSize = 8
                valSetIds = np.random.randint(low=1, high=validationDataset.__len__(), size=valBatchSize)

                # Make a concatenated tensor of inputs
                for i in range(valBatchSize):
                    valData = validationDataset[valSetIds[i]]
                    usValIn = torch.cat((usValIn, valData['us'].unsqueeze(0)), 0)
                    mriValIn = torch.cat((mriValIn, valData['mri'].unsqueeze(0)), 0)

                usValIn = usValIn.to(device)
                mriValIn = mriValIn.to(device)

                mriValOut = mriForwardmodel(mriValIn)
                usValOut = usForwardmodel(usValIn)
                #StdMean Normalization, later Sigmoid assures the range of the image is between 0-1
                #mriValOut = meanStdNormalize(mriValOut, targetMean=0.0, targetStd=1)
                #usValOut = meanStdNormalize(usValOut, targetMean=0.0, targetStd=1)
                #mriValOut = rangeNormalize(mriValOut)
                #usValOut = rangeNormalize(usValOut)

                # Compute the encoded Representations
                (_, _, _, _, mriValOutEncoded) = mriEncoderModel(mriShallowModel(mriValOut), get_features = True)
                (_, _, _, _, usValOutEncoded) = usEncoderModel(usShallowModel(usValOut), get_features = True)
                (_, _, _, _, mriValEncoded) = mriEncoderModel(mriValIn, get_features = True)
                (_, _, _, _, usValEncoded) = usEncoderModel(usValIn, get_features = True)

                # find and plot loss on the validation batch also Calculate Losses
                (totalLossVal, mriUsValOutDiffLoss, _, _, _) = calculateTotalLoss(mriValOut, usValOut, mriValOutEncoded, usValOutEncoded, mriValEncoded, usValEncoded, l2Loss, l1Loss)
                logger.appendLine(X=np.array([iteration]), Y=np.array([mriUsValOutDiffLoss.item()]), win="test", name="[VAL] MRI US Diff Loss", xlabel="Iteration Number", ylabel="Loss")
                logger.appendLine(X=np.array([iteration]), Y=np.array([totalLossVal.item()]), win="test", name="[VAL] Total Loss", xlabel="Iteration Number", ylabel="Loss")

                mriDisplayImages = torch.cat((mriValIn, mriValOut), 0)
                usDisplayImages = torch.cat((usValIn, usValOut), 0)

                mriTrainDisplayImages = torch.cat((mri, mriOut), 0)
                usTrainDisplayImages = torch.cat((us, usOut), 0)

                # plot images in visdom in a grid, MRI and US has different grids.
                logger.plotImages(images=mriDisplayImages, nrow=valBatchSize, win="mriGrid", caption='MRI Validation')
                logger.plotImages(images=usDisplayImages, nrow=valBatchSize, win="usGrid", caption='US Validation')
                logger.plotImages(images=mriTrainDisplayImages, nrow=batchSize, win="trainMRI", caption='MRI Train')
                logger.plotImages(images=usTrainDisplayImages, nrow=batchSize, win="trainUS", caption='US Train')
                
                #switch back to train mode
                usForwardmodel.train()
                mriForwardmodel.train()
        
        if epochNum%3 == 0:
            torch.save(mriForwardmodel.state_dict(), os.path.join(modelPath, "mriForward" + str(epochNum) + ".pth"))
            torch.save(usForwardmodel.state_dict(), os.path.join(modelPath, "usForward" + str(epochNum) + ".pth"))

        #Learning rate decay after every Epoch
        scheduler.step()

if __name__ == '__main__':

    parameters = ['mriEncoderPath', 'usEncoderPath']

    if len(sys.argv) == 1:
        print("Using default values")
        main()

    if len(sys.argv) > 1:

        userArgs = [s.split("=") for s in sys.argv]

        args = {}

        for i in range(1, len(userArgs)):

            if userArgs[i][0] not in parameters:
                print(userArgs[i][0] + " is not an allowed parameter. Please use one of the following:")
                print(parameters)
                sys.exit(0)

            if len(userArgs[i]) != 2:
                print("Please don't use spaces when defining a parameter. E.g. numEpochs=10")
                sys.exit(0)

            args[userArgs[i][0]] = (userArgs[i][1])

        main(**args)