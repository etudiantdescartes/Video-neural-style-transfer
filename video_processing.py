import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2 as cv
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from tqdm import tqdm
from torchvision.utils import flow_to_image



def get_frames(src):
    """
    reads a video and returns a list of images as tensors
    """
    frames = []
    cap = cv.VideoCapture(cv.samples.findFile(src))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (512, 512))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = transforms.ToTensor()(frame)
        frames.append(frame)
    cap.release()
    return frames

def get_optical_flow(frames):
    """
    Computes optical flow with pretrained RAFT model
    """
    print('Computing optical flow...')
    flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
    flow_model = flow_model.to('cuda').eval()
    transform = transforms.Normalize(mean=0.5, std=0.5)
    flow = []
    flow_video = []
    with torch.no_grad():
        for i in tqdm(range(0, len(frames)-1)):
            frame1 = transform(frames[i]).unsqueeze(0)
            frame2 = transform(frames[i+1]).unsqueeze(0)
            pred = flow_model(frame1.to('cuda'), frame2.to('cuda'))[0]
            flow.append(pred)

            flow_frame = flow_to_image(pred).squeeze(0)
            PILImage = transforms.ToPILImage()(flow_frame)
            flow_video.append(PILImage)

    return flow, flow_video

def warp_frame_with_flow(frame1, flow):
    """
    warp the frame for the initilization of the next frame and to compute temporal loss
    """
    B, C, H, W = frame1.size()

    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=frame1.device), torch.arange(W, device=frame1.device), indexing='ij')
    grid_x = (2.0 * grid_x / (W - 1)) - 1.0
    grid_y = (2.0 * grid_y / (H - 1)) - 1.0

    #optical flow displacements
    flow_x, flow_y = flow[0, 0, :, :], flow[0, 1, :, :]
    flow_x = 2.0 * flow_x / (W - 1)
    flow_y = 2.0 * flow_y / (H - 1)

    grid_x_new = grid_x + flow_x
    grid_y_new = grid_y + flow_y
    grid_new = torch.stack((grid_x_new, grid_y_new), dim=2)
    warped_frame = F.grid_sample(frame1, grid_new.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True)

    return warped_frame.squeeze(0)