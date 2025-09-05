# Sleeping Student Detector (Classroom AI Project)

This repository contains a student sleep detection project powered by **MMAction2**, designed for classroom monitoring and research applications.

## Video Sleep Classification

The model processes classroom video streams (such as lecture recordings or live webcam feeds) and classifies them as **sleeping** or **not sleeping**. The decision is based on the class with the **highest probability score**. When a sleeping event is detected, the system records the timestamp for further analysis.

## Demo

Here is an example output of the system:

![Results](Results.PNG)

## Requirements

- Python 3.8+  
- PyTorch  
- MMAction2  
- OpenCV  
- Other dependencies listed in `requirements.txt`

## Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git

2. Run the video sleep detection script:
```bash
python student.py



