# Advanced Deep Learning for Computer Vision - Multimodal Medical Image Registration

University project for Advanced Deep Learning for Computer Vision.

Topic was MRI - Ultrasound registration and the main idea was to transform both the MRI and ultrasound images in such a way that traditional registration techniques can be used to perform registration.

The main network involves two UNets with a shared decoder that transform the images and the loss is L2 loss between output images and perceptual loss between output and input images to ensure the features from the original images are present in the output. 
