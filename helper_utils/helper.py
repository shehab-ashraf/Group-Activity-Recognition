import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def show_images(imgs):
    grid_img = torchvision.utils.make_grid(imgs, nrow=len(imgs))
    npimg = grid_img.numpy()
    plt.figure(figsize=(len(imgs)*2, 2))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def show_frame(frames):
    show_images(frames)
