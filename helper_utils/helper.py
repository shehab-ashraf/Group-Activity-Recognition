import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional



def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_frame(frames):
    plt.figure(figsize=(18, 3))
    for i in range(len(frames)):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(torchvision.transforms.functional.to_pil_image(frames[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()