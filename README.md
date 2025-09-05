# Sleeping Student Detector (Classroom AI Project)

This repository contains a student sleep detection project powered by **MMAction2**, designed for classroom monitoring and research applications.

The model processes classroom video streams (such as lecture recordings or live webcam feeds) and classifies them as **sleeping** or **not sleeping** using MMAction. The decision is based on the class with the **highest probability score**. When a sleeping event is detected, the system records the timestamp for further analysis.

## Demo

Here is an example output of the system:

![Results](sleepingImage.jpg)
![Results](Results.PNG)

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










