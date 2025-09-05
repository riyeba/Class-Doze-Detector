# Sleeping Student Detector (Classroom AI Project)

This repository contains a student sleep detection project powered by **MMAction2**, designed for classroom monitoring and research applications.

The model processes classroom video streams (such as lecture recordings or live webcam feeds) and classifies them as **sleeping** or **not sleeping** using MMAction. The decision is based on the class with the **highest probability score**. When a sleeping event is detected, the system records the timestamp for further analysis.

## Demo

Here is an example output of the system:

![Results](sleepingImage.jpg)
![Results](Results.PNG)


## Features

- Real-time detection of sleeping students in classroom videos  
- Processes both recorded lectures and live webcam streams  
- Records timestamps of detected sleeping events for analysis  
- Easy-to-use setup with Python and MMAction2  

## Applications

- Classroom monitoring and student engagement analysis  
- Research on attention and sleep patterns in educational settings  
- Automated attendance and alert systems  
- Integration with lecture recording platforms or AI classroom assistants  


## Requirements

- Python 3.8+  
- PyTorch  
- MMAction2  
- OpenCV  
- torchvision

## Installation

1. **Install mmcv:**
```bash
pip install mmcv-full==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
```

2. **Clone the repository:**
```bash
git clone https://github.com/riyeba/Class-Doze-Detector.git
```

## Pre-trained Model

The trained **MMAction2** model checkpoint exported from Google Colab is available on Google Drive:  

[Download from Google Drive](https://drive.google.com/drive/folders/1_9qtJNLwtWeY1eeqiwECoBQb_9T4ftUY?usp=sharing)


## Usage

After installing the required packages and downloading the pre-trained model, you can run the detector on a video file or a live webcam feed.

### Example: Run on a video file

```python
from mmaction.apis import init_recognizer, inference_recognizer

# Paths to your config and trained model
config_file = 'configs/recognition/tsn/tsn_r50_video.py'
checkpoint_file = 'checkpoints/sleeping_student.pth'

# Initialize the model
model = init_recognizer(config_file, checkpoint_file, device='cuda:0')  # or 'cpu'

# Run inference on a classroom video
video_path = 'videos/classroom.mp4'
result = inference_recognizer(model, video_path)

# Print the result
print("Prediction:", result)











