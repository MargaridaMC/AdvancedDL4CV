import sys
import os
import sintefDataset as dataset
import matplotlib.pyplot as plt
import torch as torch
import torch.utils.data as batchloader
import perceptualModel as unetModel
import visdomLogger as vl
import numpy as np
import gc

def main(numEpochs = 100, batchSize = 5, learningRate = 1e-3):

    numEpochs, batchSize = int(numEpochs), int(batchSize)

    # Use GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # create visdom logger instance
    logger = vl.VisdomLogger(port="65531")
    logger.deleteWindow(win="test")

    # Initialize the MRI US dataset and the dataset loader
    dir = os.path.dirname(__file__)
    trainDatasetDir = os.path.join(dir, '..', 'dataset/sintefHdf5Volumes/train')
    trainDataset = dataset.SintefDataset(rootDir=trainDatasetDir)

    valDatasetDir = os.path.join(dir, '..', 'dataset/sintefHdf5Volumes/val')
    validationDataset = dataset.SintefDataset(rootDir=valDatasetDir)

    dataloader = batchloader.DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=1)

    modelPath = os.path.join(dir, '..', 'savedModels/sintef')

    # (to change back to unet just replace AutoEncoder with UNet)
    mriForwardmodel  = unetModel.AutoEncoder(n_classes=1, n_channels=1).to(device)
    usForwardmodel   = unetModel.AutoEncoder(n_classes=1, n_channels=1).to(device)

    networkParams = list(mriForwardmodel.parameters()) + list(usForwardmodel.parameters())

    lossCriteria = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(networkParams, lr = learningRate)#SGD(networkParams, lr=learningRate)

    iteration = 0
    for epochNum in range(numEpochs):
        for i_batch, sample_batched in enumerate(dataloader):
            mri = sample_batched['mri'].to(device)
            us = sample_batched['us'].to(device)

            optimizer.zero_grad()

            # Compute the shared representation
            [mriOut, usOut] = mriForwardmodel(mri, us)

            # Compute loss
            mriLoss = lossCriteria(mriOut, mri)
            usLoss = lossCriteria(usOut, us)
            totalLoss = mriLoss + usLoss

            # print Epoch number and Loss
            print("Epoch: ", epochNum, "iteration: ", i_batch, "Total loss: ", totalLoss.item(),
                  "MriUSLoss: ", mriLoss.item(), "US Recon Loss: ", usLoss.item())

            iteration = iteration + 1
            logger.appendLine(X=np.array([iteration]), Y=np.array([totalLoss.item()]), win="test", name="Total Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([mriLoss.item()]), win="test", name="MRI Loss", xlabel="Iteration Number", ylabel="Loss")
            logger.appendLine(X=np.array([iteration]), Y=np.array([usLoss.item()]), win="test", name="US Loss", xlabel="Iteration Number", ylabel="Loss")
            
            # perform backward pass
            totalLoss.backward()

            # perform optimization
            optimizer.step()

            if iteration%20 == 0:
                usValIn = torch.Tensor([])
                mriValIn = torch.Tensor([])

                #switch model to eval mode for validation
                mriForwardmodel.eval()
                usForwardmodel.eval()

                # Show output on a validation image after some iterations
                valSetIds = np.random.randint(low=1, high=validationDataset.__len__(), size=5)

                # Make a concatenated tensor of inputs
                for i in range(5):
                    valData = validationDataset[valSetIds[i]]
                    usValIn = torch.cat((usValIn, valData['us'].unsqueeze(0)), 0)
                    mriValIn = torch.cat((mriValIn, valData['mri'].unsqueeze(0)), 0)

                usValIn = usValIn.to(device)
                mriValIn = mriValIn.to(device)

                mriOut = mriForwardmodel(mriValIn)
                usOut = usForwardmodel(usValIn)

                mriDisplayImages = torch.cat((mriValIn, mriOut), 0)
                usDisplayImages = torch.cat((usValIn, usOut), 0)

                # plot images in visdom in a grid, MRI and US has different grids.
                logger.plotImages(images=mriDisplayImages, nrow=5, win="mriGrid", caption='MRI Validation')
                logger.plotImages(images=usDisplayImages, nrow=5, win="usGrid", caption='US Validation')

                # calculate losses
                zeroTensor = torch.cuda.FloatTensor(mriValIn.size()).fill_(0)
                mriLoss = lossCriteria(mriOut - mriValIn, zeroTensor)
                usLoss = lossCriteria(usOut - usValIn, zeroTensor)
                totalLoss = mriLoss + usLoss

                #plot validation loss
                logger.appendLine(X=np.array([iteration]), Y=np.array([totalLoss.item()]), win="test", name="Validation Loss", xlabel="Iteration Number", ylabel="Loss")

                #switch back to train mode
                mriForwardmodel.train()
                usForwardmodel.train()

        torch.save(mriForwardmodel.state_dict(), os.path.join(modelPath, "mriPretrainedModel" + str(epochNum) + ".pth"))
        torch.save(usForwardmodel.state_dict(), os.path.join(modelPath, "usPretrainedModel" + str(epochNum) + ".pth"))

if __name__ == '__main__':

    parameters = ['numEpochs', 'batchSize', 'learningRate']

    if len(sys.argv) == 1:
        print("Using default values")
        main()

    if len(sys.argv) > 1:

        userArgs = [s.split("=") for s in sys.argv]

        args = {}

        for i in range(1, loggeren(userArgs)):

            if userArgs[i][0] not in parameters:
                print(userArgs[i][0] + " is not an allowed parameter. Please use one of the following:")
                print(parameters)
                sys.exit(0)

            if len(userArgs[i]) != 2:
                print("Please don't use spaces when defining a parameter. E.g. numEpochs=10")
                sys.exit(0)

            args[userArgs[i][0]] = float(userArgs[i][1])

        main(**args)
