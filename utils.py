import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2 as cv
import glob
from natsort import natsorted, ns
import os


def save_frame(output, dst, i):
    image = output.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    save_path = os.path.join(dst, f'frame_{i}.jpg')
    image.save(save_path)

def imshow(tensor, title=None):
    """
    Convert frame to PIL before displaying it
    """
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def image_loader(image_name, device):
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0).to(device, torch.float)

    return image

def save_flow(flow):
    for i, f in enumerate(flow):
        f.save(f'frames/flow_{i}.jpg')

def video_creation(src='frames', dst='output_video.mp4'):
    """
    reads a folder of images to create a video
    """
    fps = 30
    frame_size = (512, 512)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video = cv.VideoWriter(dst, fourcc, fps, frame_size)

    for image_file in natsorted(glob.glob(f'{src}/*.jpg'), alg=ns.IGNORECASE):
        img = cv.imread(image_file)
        video.write(img)

    video.release()