import torch
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights
import os
import torch.optim as optim

from utils import save_frame, imshow, image_loader, video_creation
from model import get_style_model_and_losses
from video_processing import get_frames, get_optical_flow, warp_frame_with_flow



def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, content_img, style_img1, style_img2, input_img, previous_frame, style_weights, num_steps=200,
                       style_weight=1e5, content_weight=1, temporal_weight=1):

    model, style_losses, content_losses, temporal_losses = get_style_model_and_losses(cnn, style_img1, style_img2, content_img, style_weights, previous_frame, temporal_weight)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)


    run = [0]
    while run[0] <= num_steps:

        def closure():

            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)


            style_score = sum(sl.loss * weight for sl, weight in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            temporal_score = sum(tl.loss for tl in temporal_losses)

            loss = style_score * style_weight + content_score * content_weight + temporal_score * temporal_weight
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f} Temporal Loss: {:4f}'.format(
                    style_score.item(), content_score.item(), temporal_score))
                print()


            #saving optimization steps to display for single image processing
            #temp_output = input_img.detach().clone()
            #temp_image = transforms.ToPILImage()(temp_output.squeeze(0))
            #temp_image.save(os.path.join('/content/drive/MyDrive/frames', f"step_{run[0]}.jpg"))

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img




if __name__ == '__main__':
    device = 'cuda'
    torch.set_default_device(device)

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to('cuda')



    ############################## PARAMETERS ##################################
    process_mode = 'video'#'video' or 'single' to process videos or one image only
    use_temporal_loss = True#used only in 'video' mode


    #Each of these parameters depend heavily on the content and style images
    style_weights = [1., 1.]#weights to use two style images ([1.,0.] will only use the first style image)
    num_steps=300#number of optimization iterations, 300 is generally good number from our tests when using content image as initialization
                 #should be higher if starting from random noise
    style_weight=1e4#weight for the style loss in general (different than style_weights)
    content_weight=1
    temporal_weight=1


    style_img1_path = "dataset/picasso.jpg"
    style_img2_path = "dataset/picasso.jpg"#use the same image as the first style image for single style processing
    content_img_path = "dataset/mountain.jpeg"
    content_video_path = "dataset/herisson.mp4"


    #save individual frames in frame_output_folder
    frame_output_folder = 'frames'

    ############################################################################




    if process_mode == 'single':
        #run style transfer for one image only

        content_img = image_loader(content_img_path, device)
        style_img1 = image_loader(style_img1_path, device)
        style_img2 = image_loader(style_img2_path, device)

        input_img = torch.randn(content_img.data.size())
        output = run_style_transfer(cnn, content_img, style_img1, style_img2, input_img, None, style_weights, num_steps=num_steps,
                       style_weight=style_weight, content_weight=content_weight, temporal_weight=temporal_weight)

        plt.figure()
        imshow(output, title='Stylized Image')
        plt.show()

    elif process_mode == 'video':
        #run style transfer for a video

        if not os.path.exists(frame_output_folder):
            os.makedirs(frame_output_folder)

        frames = get_frames(content_video_path)
        optical_flow, flow_video = get_optical_flow(frames)

        previous_frame = None#no particular processing for the first frame
        for i, (frame, flow) in enumerate(zip(frames, optical_flow)):
            print(f'Processing frame {i}')

            content_img = frame.unsqueeze(0).to(device)
            style_img1 = image_loader(style_img1_path, device)
            style_img2 = image_loader(style_img2_path, device)

            if i == 0:
                input_img = content_img.clone()#starting from the content image instead of random noise can speed up the process
                prev = None
            else:
                if use_temporal_loss and previous_frame is not None:
                    input_img = warp_frame_with_flow(previous_frame, -optical_flow[i-1].to(device)).detach().unsqueeze(0)
                    prev = input_img.clone()
                else:
                    input_img = content_img.clone()
                    prev = None

            output = run_style_transfer(cnn, content_img, style_img1, style_img2, input_img, prev, style_weights, num_steps=num_steps,
                       style_weight=style_weight, content_weight=content_weight, temporal_weight=temporal_weight)

            plt.figure()
            imshow(output, title='Output Image')
            plt.show()

            save_frame(output, 'frames', i)

            if use_temporal_loss:
                previous_frame = output.clone()

        video_creation(src='frames', dst='output_video.mp4')