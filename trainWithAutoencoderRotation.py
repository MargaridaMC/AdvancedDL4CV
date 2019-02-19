import sys
import os
import sintefDatasetValTest as dataset
import matplotlib.pyplot as plt
import torch as torch
import torch.utils.data as batchloader
import unet_model as unetModel
from unet_parts import *
from perceptualModel import AutoEncoder
from perceptualParts import *
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

def viewActivations(x, net, logger, windowName = 'x'):

    padding = 1
    ubound = 255.0

    xout = net.forward(x, get_features = True)

    for i in range(len(xout)):
        
        xi = xout[i]

        xi = xi.detach().cpu().numpy()

        (N, C, H, W) = xi.shape
        grid_height = H * C + padding * (C - 1)
        grid_width = W * N + padding * (N - 1)

        xgrid = np.zeros((grid_height, grid_width))

        x0, x1 = 0, W
        idx = 0

        while x1 <= grid_width:

            y0, y1 = 0, H
            channel = 0

            while y1 <= grid_height:
                
                ximg = xi[idx, channel, :, :]

                xlow, xhigh = np.min(ximg), np.max(ximg)
                if xhigh != xlow:
                    xgrid[y0:y1, x0:x1] = ubound * (ximg - xlow) / (xhigh - xlow)

                y0 += H + padding
                y1 += H + padding

                channel += 1

            x0 += W + padding
            x1 += W + padding

            idx += 1

        logger.plotImages(images=xgrid, nrow=10, win=windowName + str(i), caption = windowName)

def registrationLoss(rotatedMrisOut, usOut, loss, gtLoss):

    regLoss = torch.cuda.FloatTensor([0])
    for i in range(rotatedMrisOut.shape[1]):
        l2Diff = loss(rotatedMrisOut[:,i,:,:], usOut[:,0,:,:])# - gtLoss
        #print(l2Diff)
        regLoss += l2Diff
    return 1/regLoss

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

    regLoss = registrationLoss(rotatedMrisOut, usOut, l2Loss, mriUsOutDiffLoss)

    totalLoss = 8000000*mriUsOutDiffLoss + mriEnLoss + 2*usEnLoss + 50000000*regLoss
    return (totalLoss, 8000000*mriUsOutDiffLoss, mriEnLoss, 2*usEnLoss, 50000000*regLoss)

def main(mriEncoderPath, usEncoderPath):

    numEpochs, batchSize = 12, 4

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
    mriForwardmodel  = unetModel.UNet(n_classes=1, n_channels=1).to(device)
    usForwardmodel   = unetModel.UNet(n_classes=1, n_channels=1).to(device)
    networkParams = list(mriForwardmodel.parameters()) + list(usForwardmodel.parameters())
    optimizer = torch.optim.Adam(networkParams, lr=1e-4)#, weight_decay=1e-9)

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
            rotatedMris = sample_batched['rotatedMris'].to(device)

            # perform optimization
            optimizer.zero_grad()

            # Compute the shared representation
            mriOut = mriForwardmodel(mri)
            usOut = usForwardmodel(us)

            rotatedMrisOut = mriForwardmodel(rotatedMris.view(rotatedMris.shape[0] * rotatedMris.shape[1], 1, rotatedMris.shape[2], rotatedMris.shape[3]))
            rotatedMrisOut = rotatedMrisOut.view(rotatedMris.shape[0], rotatedMris.shape[1], rotatedMris.shape[2], rotatedMris.shape[3])

            # Compute the encoded Representations
            (_, mriOutEncoded, _, _) = mriEncoderModel(mriOut, get_features = True)
            (_, usOutEncoded, _, _) = usEncoderModel(usOut, get_features = True)
            (_, mriEncoded, _, _) = mriEncoderModel(mri, get_features = True)
            (_, usEncoded, _, _) = usEncoderModel(us, get_features = True)

             # Calculate Losses
            (totalLoss, mriUsOutDiffLoss, mriEnLoss, usEnLoss, regLoss) = calculateTotalLoss(mriOut, usOut, rotatedMrisOut, mriOutEncoded, usOutEncoded, mriEncoded, usEncoded, l2Loss, l1Loss)


            # perform backward pass
            totalLoss.backward()

            optimizer.step()

            # print Epoch number and Loss
            print("Epoch: ", epochNum, "iteration: ", i_batch, "Total loss: ", totalLoss.item(),
                  "MriUSLoss: ", mriUsOutDiffLoss.item(), "MriEncoderLoss: ", mriEnLoss.item(), "UsEncoderLoss:", usEnLoss.item(), "RegistrationLoss", regLoss.item())

            iteration = iteration + 1
            logger.appendLine(X=np.array([iteration]), Y=np.array([totalLoss.item()]), win="test", name="Total Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([mriUsOutDiffLoss.item()]), win="test", name="MRI US Diff Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([mriEnLoss.item()]), win="test", name="MRI Encoder Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([usEnLoss.item()]), win="test", name="US Encoder Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([regLoss.item()]), win="test", name="Registration Loss", xlabel="Iteration Number", ylabel="Loss")

            if iteration%20 == 0:
                usValIn = torch.Tensor([])
                mriValIn = torch.Tensor([])
                rotatedMrisValIn = torch.Tensor([])
                
                #switch model to eval mode for validation
                mriForwardmodel.eval()
                usForwardmodel.eval()

                # Show output on a validation image after some iterations
                valBatchSize = 5
                valSetIds = np.random.randint(low=1, high=validationDataset.__len__(), size=valBatchSize)

                # Make a concatenated tensor of inputs
                for i in range(valBatchSize):
                    valData = validationDataset[valSetIds[i]]
                    usValIn = torch.cat((usValIn, valData['us'].unsqueeze(0)), 0)
                    mriValIn = torch.cat((mriValIn, valData['mri'].unsqueeze(0)), 0)
                    rotatedMrisValIn = torch.cat((rotatedMrisValIn, valData['rotatedMris'].unsqueeze(0)), 0)

                usValIn = usValIn.to(device)
                mriValIn = mriValIn.to(device)
                rotatedMrisValIn = rotatedMrisValIn.to(device)

                mriValOut = mriForwardmodel(mriValIn)
                usValOut = usForwardmodel(usValIn)
                
                rotatedMrisValOut = mriForwardmodel(rotatedMrisValIn.view(rotatedMrisValIn.shape[0] * rotatedMrisValIn.shape[1], 1, rotatedMrisValIn.shape[2], rotatedMrisValIn.shape[3]))
                rotatedMrisValOut = rotatedMrisValOut.view(rotatedMrisValIn.shape[0], rotatedMrisValIn.shape[1], rotatedMrisValIn.shape[2], rotatedMrisValIn.shape[3])               

                # Compute the encoded Representations
                (_, mriValOutEncoded, _, _) = mriEncoderModel(mriValOut, get_features = True)
                (_, usValOutEncoded, _, _) = usEncoderModel(usValOut, get_features = True)
                (_, mriValEncoded, _, _) = mriEncoderModel(mriValIn, get_features = True)
                (_, usValEncoded, _, _) = usEncoderModel(usValIn, get_features = True)

                #viewActivations(mriValIn, mriEncoderModel, logger, "MRI In Activations")
                #viewActivations(usValIn, usEncoderModel, logger, "US In Activations")
                #viewActivations(mriValOut, mriEncoderModel, logger, "MRI Out Activations")
                #viewActivations(usValOut, usEncoderModel, logger, "US Out Activations")

                """
                print('')
                print("MRI running mean ", mriForwardmodel.batchNorm.running_mean)
                print('')
                print("US running mean ", usForwardmodel.batchNorm.running_mean)
                print('')
                print("MRI running var ", mriForwardmodel.batchNorm.running_var)
                print('')
                print("US running var ", usForwardmodel.batchNorm.running_var)
                print('')
                """

                # find and plot loss on the validation batch also Calculate Losses

                (totalLossVal, mriUsValOutDiffLoss, _, _, _) = calculateTotalLoss(mriValOut, usValOut, rotatedMrisValOut, mriValOutEncoded, usValOutEncoded, mriValEncoded, usValEncoded, l2Loss, l1Loss)
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
        
            if epochNum%1 == 0:
                print("Saved model ", iteration)
                torch.save(mriForwardmodel.state_dict(), os.path.join(modelPath, "mriForward" + str(epochNum) + ".pth"))
                torch.save(usForwardmodel.state_dict(), os.path.join(modelPath, "usForward" + str(epochNum) + ".pth"))

        #Learning rate decay after every Epoch
        optimizer.step()

    print("Training ended")

if __name__ == '__main__':

    main(mriEncoderPath = "../savedModels/autoencoder/mriAutoencoder4.pth",usEncoderPath = "../savedModels/autoencoder/usAutoencoder4.pth")

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