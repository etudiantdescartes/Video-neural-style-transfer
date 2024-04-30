# Video-neural-style-transfer
This project is an attempt at adapting a basic neural style transfer algorithm for videos by enforcing temporal consistency within consecutive frames.
There are different ways of achieving style transfer:
- By training models such as GANs, which requires a lot of training data and material resources
- By optimizing an image to make it look like a content image and a loss image. The optimization takes a bit of time but it doesn't require training.

In this project I used the optimization-based method.

The method involves optimizing an image initially made up of noise to make it resemble a style image and a content image. To do this, we pass the three images through the VGG-19 model to retrieve relevant features and calculate the loss functions associated with style and content (see the following image). For content, we use the features from the conv_4 layer, which is relatively deep in the model, allowing us to capture higher-level features, i.e., the general information of the scene. For style, we retrieve features from layers conv_1 to conv_5, which contain general information as well as the texture details of the image. We then calculate the content loss using mean square error (MSE), and the style loss by computing the Gram matrix, followed by MSE.

![image1](/2.png?raw=true)

In the following example, I attempted to apply the style of a painting from Picasso the a video of a hedgehog. As we can see here, the stylized video has a lot of flickering in the background, due to the fact that each frame is stylized individually.

<div style="text-align: center;">
  <img src="picasso.jpg?raw=true" alt="Style image" style="width: 300px; height: 300px; display: inline-block;">
  <img src="hedgehog.gif?raw=true" alt="Original hedgehog video" style="width: 300px; height: 300px; display: inline-block;">
  <img src="style1.gif?raw=true" alt="Stylized video" style="width: 300px; height: 300px; display: inline-block;">
</div>

To address this issue, we compute the optical flow of the video using the pre-trained RAFT model. For each frame, we warp the previous one with the corresponding optical flow to get a starting point close to the final result of the current frame. This helps to reduce inconsistencies between frames and speeds up image processing. We then pass the current frame and the warped previous frame through the feature extractor, and calculate the MSE for conv_1 and conv_2 layers between the two frames to optimize the current frame. This produces frames with less variation in style, by forcing the image to not deviate too much from its initial state.

Here are the results:

<div style="text-align: center;">
  <img src="style1.gif?raw=true" alt="Stylized video without temporal consistency" style="width: 300px; height: 300px; display: inline-block;">
  <img src="style2.gif?raw=true" alt="Stylized video with temporal consistency" style="width: 300px; height: 300px; display: inline-block;">
</div>


# Installation

## Prerequisites
First install torch, torchvision and opencv:
- ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```
- ```pip install opencv-python```
- Only works with a GPU

## Running the code
The paths and parameters are hardcoded so make sure you have a folder called "frames" at the root of the project before runing ```main.py```, and specify the parameters inside it.
