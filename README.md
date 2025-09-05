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

## Usage

### Run on a video

```python
from mmaction.apis import init_recognizer, inference_recognizer
from operator import itemgetter

# Paths
config_file = 'configs/recognition/tsn/tsn_sleeping.py'
checkpoint_file = 'work_dirs/tsn_sleeping/best_acc_top1_epoch_1.pth'
video_file = 'videos/classroom.mp4'
label_file = 'label_map.txt'

# Initialize model
model = init_recognizer(config_file, checkpoint_file, device='cpu')  # or 'cuda:0'

# Run inference
pred_result = inference_recognizer(model, video_file)

# Process results
pred_scores = pred_result.pred_score.tolist()
score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
top5_label = score_sorted[:5]

# Load label names
with open(label_file) as f:
    labels = [x.strip() for x in f.readlines()]

# Display top predictions
for idx, score in top5_label:
    print(f"{labels[idx]}: {score:.4f}")

```

## Acknowledgements

We would like to thank the developers of [MMAction2](https://github.com/open-mmlab/mmaction2) and [MMCV](https://github.com/open-mmlab/mmcv) for providing the powerful video understanding framework that made this project possible.  

We also acknowledge the contributions of the open-source community and the tools used in this project, including PyTorch and OpenCV.










