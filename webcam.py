#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
from skimage.transform import resize as imresize
import cv2
import torch
from torchvision import models
import torchvision.transforms as transforms
from cae import SegNet_Classifier
from matplotlib.colors import hsv_to_rgb

INPUT_SIZE = (32, 32)  # (224, 224) for models pre-trained on ImageNet

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

inference_transformation = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(cinic_mean, cinic_std),
])


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
Select device
'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Requires PyTorch 0.4 or higher

'''
Load trained net
'''
net = SegNet_Classifier(10)
net.load_state_dict(torch.load('segnet_classification_50_4.pt'))
#net.to(device)
net.eval()

cap = cv2.VideoCapture(-1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess frame
    img = Image.fromarray(frame)
    img_tensor = inference_transformation(img).unsqueeze_(0)
    #img_tensor.to(device)

    # Prediction step
    _hsv_im, _, _pred, _prob, _hsv_im_l, _attn = net.segment(img_tensor)

    gs_im = (np.array(img)/255).mean(-1)
    def _resize(hsv_im):
        hue = imresize(hsv_im[:,:,0], gs_im.shape)
        sat = imresize(hsv_im[:,:,1], gs_im.shape)
        im = hsv_to_rgb(np.stack([hue, sat, gs_im], axis=-1))
        return im

    pred = _pred.detach().numpy()
    im = hsv_to_rgb(_hsv_im.detach().numpy())[0]
    im = _resize(im)
    im -= im.min()
    im /= im.max()


    # Display the resulting frame
    cv2.putText(img=im, text=classes[pred],
                org=(150, 75),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=4, color=(255, 255, 255))
    cv2.imshow('frame', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
