import sys
import os
import sintefDatasetValTest as dataset
import matplotlib.pyplot as plt
import torch as torch
import torch.utils.data as batchloader
from ynet_model import *
from unet_parts import *
from AperceptualModel import AutoEncoder
from AperceptualParts import *
import visdomLogger as vl
import numpy as np
import gc
import torchvision

# Constants
dx = torch.cuda.FloatTensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]]).detach()

dy = torch.cuda.FloatTensor([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]]).detach()
dx = dx.view((1,1,3,3))
dy = dy.view((1,1,3,3))

def registrationLoss(rotatedMrisOut, usOut, loss):
    zeroTensor = torch.cuda.FloatTensor(rotatedMrisOut.size()).fill_(0).detach()
    regLoss = loss(rotatedMrisOut - usOut.expand_as(rotatedMrisOut), zeroTensor)
    regLoss = 1/(regLoss + 1e-8)
    return regLoss

def gradient(image):
    G_x = torch.nn.functional.conv2d(image, dx)
    G_y = torch.nn.functional.conv2d(image, dy)

    return torch.sqrt(torch.pow(G_x, 2)+ torch.pow(G_y, 2))

def calculateTotalLoss(mriOut, usOut, rotatedMrisOut, mriOutEncoded, usOutEncoded, mriEncoded, usEncoded, l2Loss, l1Loss):
    # calculate losses
    zeroTensor = torch.cuda.FloatTensor(mriOut.size()).fill_(0).detach()
    mriUsOutDiffLoss = l2Loss(mriOut - usOut, zeroTensor)

    #zeroTensor2 = torch.cuda.FloatTensor(mriOutEncoded.size()).fill_(0).detach()
    mriEnLoss = l2Loss(mriOutEncoded, mriEncoded)
    usEnLoss = l2Loss(usOutEncoded, usEncoded)

    #regLoss = registrationLoss(rotatedMrisOut, usOut, l2Loss)

    totalLoss = 10000000*mriUsOutDiffLoss + 10*mriEnLoss + usEnLoss# + 50*regLoss
    return (totalLoss, 10000000*mriUsOutDiffLoss, 10*mriEnLoss, usEnLoss, 0)

def main(mriEncoderPath, usEncoderPath):

    numEpochs, batchSize = 5, 4

    # Use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # create visdom logger instance
    logger = vl.VisdomLogger(port="65531")
    logger.deleteWindow(win="test")

    # Initialize the MRI US dataset and the dataset loader
    dir = os.path.dirname(__file__)
    modelPath = os.path.join(dir, '..', 'savedModels/transformation')

    trainDatasetDir = '../dataset/sintefHdf5Volumes/train'
    trainDataset = dataset.SintefDatasetValTest(rootDir=trainDatasetDir, mriFilePrefix='T1-Vol', usFilePrefix='US-DS')

    valDatasetDir = '../dataset/sintefHdf5Volumes/val'
    validationDataset = dataset.SintefDatasetValTest(rootDir=valDatasetDir, mriFilePrefix='T1-Vol', usFilePrefix='US-DS')

    dataloader = batchloader.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    
    #create transformation models and add them to optimizer
    mriForwardmodel  = Encoder(n_classes=1, n_channels=1).to(device)
    usForwardmodel   = Encoder(n_classes=1, n_channels=1).to(device)
    commonBranch     = Decoder(n_classes=1, n_channels=1).to(device)


    networkParams = list(mriForwardmodel.parameters()) + list(usForwardmodel.parameters()) + list(commonBranch.parameters())
    optimizer = torch.optim.Adam(networkParams, lr=1e-3, weight_decay=1e-9)

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
    for epochNum in range(numEpochs):
        for i_batch, sample_batched in enumerate(dataloader):
            mri = sample_batched['mri'].to(device)
            us = sample_batched['us'].to(device)
            #rotatedMris = sample_batched['rotatedMris'].to(device)

            # perform optimization
            optimizer.zero_grad()

            # Compute the shared representation
            mriFeatures = mriForwardmodel(mri)
            usFeatures = usForwardmodel(us)
            mriOut = commonBranch(mriFeatures)
            usOut = commonBranch(usFeatures)

            #rotatedMrisOut = mriForwardmodel(rotatedMris.view(rotatedMris.shape[0] * rotatedMris.shape[1], 1, rotatedMris.shape[2], rotatedMris.shape[3]))
            #rotatedMrisOut = rotatedMrisOut.view(rotatedMris.shape[0], rotatedMris.shape[1], rotatedMris.shape[2], rotatedMris.shape[3])

            # Compute the encoded Representations
            (mriOutEncoded1, mriOutEncoded2, mriOutEncoded3, mriOutEncoded4) = mriEncoderModel(mriOut, get_features = True)
            (usOutEncoded1, usOutEncoded2, usOutEncoded3, usOutEncoded4) = usEncoderModel(usOut, get_features = True)
            (mriEncoded1, mriEncoded2, mriEncoded3, mriEncoded4) = mriEncoderModel(mri, get_features = True)
            (usEncoded1, usEncoded2, usEncoded3, usEncoded4) = usEncoderModel(us, get_features = True)

            mriEnLoss = (l2Loss(mriOutEncoded1, mriEncoded1) + l2Loss(mriOutEncoded2, mriEncoded2) + l2Loss(mriOutEncoded3, mriEncoded3) + l2Loss(mriOutEncoded4, mriEncoded4))/4000
            usEnLoss = (l2Loss(usOutEncoded1, usEncoded1) + l2Loss(usOutEncoded2, usEncoded2) + l2Loss(usOutEncoded3, usEncoded3) + l2Loss(usOutEncoded4, usEncoded4))/4000

             # Calculate Losses
            #(totalLoss, mriUsOutDiffLoss, mriEnLoss, usEnLoss, _) = calculateTotalLoss(mriOut, usOut, _, mriOutEncoded, usOutEncoded, mriEncoded, usEncoded, l2Loss, l1Loss)

            featureLoss = 100000*l2Loss(mriFeatures[-1] - usFeatures[-1], torch.cuda.FloatTensor(mriFeatures[-1].size()).fill_(0))
            #print(len(mriFeatures))
            #for i in len(mriFeatures):
            # 	zeroTensor = torch.cuda.FloatTensor(mriFeatures[i].size()).fill_(0)
            #	featureLoss += l2Loss(mriFeatures[i] - usFeatures[i], zeroTensor)


            zeroTensor = torch.cuda.FloatTensor(mriOut.size()).fill_(0)
            mriUsOutDiffLoss = 1000000*l2Loss(mriOut - usOut, zeroTensor)
            #mriEnLoss = l2Loss(mriOut, mri)
            #usEnLoss = l2Loss(usOut, us)

            totalLoss = featureLoss + mriUsOutDiffLoss + mriEnLoss + usEnLoss

            # perform backward pass
            totalLoss.backward()

            optimizer.step()

            # print Epoch number and Loss
            print("Epoch: ", epochNum, "iteration: ", i_batch, "Total loss: ", totalLoss.item(),
                  "MriUSLoss: ", mriUsOutDiffLoss.item(), "MriEncoderLoss: ", mriEnLoss.item(), "UsEncoderLoss:", usEnLoss.item(), "featureLoss", featureLoss.item())

            iteration = iteration + 1
            logger.appendLine(X=np.array([iteration]), Y=np.array([totalLoss.item()]), win="test", name="Total Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([featureLoss.item()]), win="test", name="Feature Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([mriUsOutDiffLoss.item()]), win="test", name="MRI US Diff Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([mriEnLoss.item()]), win="test", name="MRI Encoder Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([usEnLoss.item()]), win="test", name="US Encoder Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([featureLoss.item()]), win="test", name="Feature Loss", xlabel="Iteration Number", ylabel="Loss")

            if iteration%20 == 0:
                usValIn = torch.Tensor([])
                mriValIn = torch.Tensor([])
                #rotatedMrisValIn = torch.Tensor([])
                
                #switch model to eval mode for validation
                mriForwardmodel.eval()
                usForwardmodel.eval()
                commonBranch.eval()

                # Show output on a validation image after some iterations
                valBatchSize = 5
                valSetIds = np.random.randint(low=1, high=validationDataset.__len__(), size=valBatchSize)

                # Make a concatenated tensor of inputs
                for i in range(valBatchSize):
                    valData = validationDataset[valSetIds[i]]
                    usValIn = torch.cat((usValIn, valData['us'].unsqueeze(0)), 0)
                    mriValIn = torch.cat((mriValIn, valData['mri'].unsqueeze(0)), 0)
                    #rotatedMrisValIn = torch.cat((rotatedMrisValIn, valData['rotatedMris'].unsqueeze(0)), 0)

                usValIn = usValIn.to(device)
                mriValIn = mriValIn.to(device)
                #rotatedMrisValIn = rotatedMrisValIn.to(device)

                mriValFeatures = mriForwardmodel(mriValIn)
                usValFeatures = usForwardmodel(usValIn)
                mriValOut = commonBranch(mriValFeatures)
                usValOut = commonBranch(usValFeatures)
                
                #rotatedMrisValOut = mriForwardmodel(rotatedMrisValIn.view(rotatedMrisValIn.shape[0] * rotatedMrisValIn.shape[1], 1, rotatedMrisValIn.shape[2], rotatedMrisValIn.shape[3]))
                #rotatedMrisValOut = rotatedMrisValOut.view(rotatedMrisValIn.shape[0], rotatedMrisValIn.shape[1], rotatedMrisValIn.shape[2], rotatedMrisValIn.shape[3])               

                # Compute the encoded Representations
                (mriValOutEncoded1, mriValOutEncoded2, mriValOutEncoded3, mriValOutEncoded4) = mriEncoderModel(mriValOut, get_features = True)
                (usValOutEncoded1, usValOutEncoded2, usValOutEncoded3, usValOutEncoded4) = usEncoderModel(usValOut, get_features = True)
                (mriValEncoded1, mriValEncoded2, mriValEncoded3, mriValEncoded4) = mriEncoderModel(mriValIn, get_features = True)
                (usValEncoded1, usValEncoded2, usValEncoded3, usValEncoded4) = usEncoderModel(usValIn, get_features = True)

                mriEnLoss = (l2Loss(mriValOutEncoded1, mriValEncoded1) + l2Loss(mriValOutEncoded2, mriValEncoded2) + l2Loss(mriValOutEncoded3, mriValEncoded3)) + l2Loss(mriValOutEncoded4, mriValEncoded4)/4000
                usEnLoss = (l2Loss(usValOutEncoded1, usValEncoded1) + l2Loss(usValOutEncoded2, usValEncoded2) + l2Loss(usValOutEncoded3, usValEncoded3) + l2Loss(usValOutEncoded4, usValEncoded4))/4000


                # find and plot loss on the validation batch also Calculate Losses

                #(totalLossVal, mriUsValOutDiffLoss, _, _, _) = calculateTotalLoss(mriValOut, usValOut, _, mriValOutEncoded, usValOutEncoded, mriValEncoded, usValEncoded, l2Loss, l1Loss)

                featureLoss = 100000*l2Loss(mriValFeatures[-1] - usValFeatures[-1], torch.cuda.FloatTensor(mriValFeatures[-1].size()).fill_(0))

                #for i in len(mriFeatures):
                #	zeroTensor = torch.cuda.FloatTensor(mriValFeatures[i].size()).fill_(0)
                #	featureLoss += l2Loss(mriValFeatures[i] - usValFeatures[i], zeroTensor)
    
                zeroTensor = torch.cuda.FloatTensor(mriValOut.size()).fill_(0)
                mriUsValOutDiffLoss = 1000000*l2Loss(mriValOut - usValOut, zeroTensor)
                #mriEnLoss = l2Loss(mriValOut, mriValIn)
                #usEnLoss = l2Loss(usValOut, usValIn)

                totalLossVal = featureLoss + mriUsValOutDiffLoss + mriEnLoss + usEnLoss

                logger.appendLine(X=np.array([iteration]), Y=np.array([mriUsValOutDiffLoss.item()]), win="test", name="[VAL] MRI US Diff Loss", xlabel="Iteration Number", ylabel="Loss")
                logger.appendLine(X=np.array([iteration]), Y=np.array([totalLossVal.item()]), win="test", name="[VAL] Total Loss", xlabel="Iteration Number", ylabel="Loss")

                mriDisplayImages = torch.cat((mriValIn, rangeNormalize(mriValOut)[0]), 0)
                usDisplayImages = torch.cat((usValIn, rangeNormalize(usValOut)[0]), 0)

                mriTrainDisplayImages = torch.cat((mri, rangeNormalize(mriOut)[0]), 0)
                usTrainDisplayImages = torch.cat((us, rangeNormalize(usOut)[0]), 0)

                # plot images in visdom in a grid, MRI and US has different grids.
                logger.plotImages(images=mriDisplayImages, nrow=valBatchSize, win="mriGrid", caption='MRI Validation')
                logger.plotImages(images=usDisplayImages, nrow=valBatchSize, win="usGrid", caption='US Validation')
                logger.plotImages(images=mriTrainDisplayImages, nrow=batchSize, win="trainMRI", caption='MRI Train')
                logger.plotImages(images=usTrainDisplayImages, nrow=batchSize, win="trainUS", caption='US Train')
                
                #switch back to train mode
                usForwardmodel.train()
                mriForwardmodel.train()
                commonBranch.train()
        
        #if epochNum%1 == 0:
        #    torch.save(mriForwardmodel.state_dict(), os.path.join(modelPath, "mriYNet" + str(epochNum) + ".pth"))
        #    torch.save(usForwardmodel.state_dict(), os.path.join(modelPath, "usYNet" + str(epochNum) + ".pth"))
        #    torch.save(commonBranch.state_dict(), os.path.join(modelPath, "commonBranchYNet" + str(epochNum) + ".pth"))

        #Learning rate decay after every Epoch
        scheduler.step()

    print("Training ended")

if __name__ == '__main__':

	main(mriEncoderPath = "../savedModels/autoencoder/AmriAutoencoder3.pth",usEncoderPath = "../savedModels/autoencoder/AusAutoencoder3.pth")
	#main()
	"""

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
        """