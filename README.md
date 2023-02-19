# Automatic Number Plate Recognition (ANPR)

### Description

The automatic number plate recognition (ANPR) system reads and recognises vehicle number plates using computer vision and image processing methods. A popular object detection model in computer vision problems is YOLOv7. Python software called EasyOCR has optical character recognition (OCR) capabilities.

Use these procedures to perform an ANPR using YOLOv7 and EasyOCR:

* Accumulate a collection of photos showing licence plates for vehicles.
* The dataset can be used to train the YOLOv7 model to recognise licence plates in the photos.
* Use EasyOCR to extract the characters from the number plates that YOLOv7 has detected.
* Extract the licence plate number by using a character recognition algorithm to identify the characters.

A combination of computer vision, machine learning, and image processing methods are needed to solve the difficult problem of ANPR. The numerous variables involved must be carefully tuned and optimised. But, you may create a reliable and accurate ANPR system utilising YOLOv7 and EasyOCR.
<div align="center">
    <img src="./figure/performance.png" width="79%"/>
</div>

## Clone our Repository
```shell
git clone https://github.com/ANPR-ORG/ANPR-using-YOLOV7-EasyOCR.git
cd ANPR-using-YOLOV7-EasyOCR
pip install -r requirements.txt
```
## Downloading Dataset
Create Directory
``` shell
mkdir custom_dataset
cd custom_dataset
```
Install dataset along with annotations using roboflow
``` shell
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="mPmElqud1pnkGfoA7Emu")
project = rf.workspace("college-dbbrk").project("anpr-x1a2o")
dataset = project.version(5).download("yolov7")
cd ..
```

## Downloading Trained Model
Creating Directory
``` shell
mkdir models
cd models
```
Downloading our Trained Model
``` shell
wget https://drive.google.com/file/d/1muS4VhL72di10Ob8-mHLmTz2cLDdz9Ug/view?usp=share_link
cd ..
```
[`best.pt`](https://drive.google.com/file/d/1muS4VhL72di10Ob8-mHLmTz2cLDdz9Ug/view?usp=share_link)

## Inferencing on Data
``` shell
cd ANPR-using-YOLOV7-EasyOCR
```
On WebCam Livefeed:
``` shell
#GPU
python detect.py --weights models/best.pt --conf 0.25 --img-size 640 --source 0 --device 0
#CPU
python detect.py --weights models/best.pt --conf 0.25 --img-size 640 --source 0 --device 'cpu'
```
On Image/Video:
``` shell
#GPU
python detect.py --weights models/best.pt --conf 0.25 --img-size 640 --source <image_path/video_path> --device 0
#CPU
python detect.py --weights models/best.pt --conf 0.25 --img-size 640 --source <image_path/video_path> --device 'cpu'
```
## Using Streamlit
``` shell
cd ANPR-using-YOLOV7-EasyOCR
streamlit run streamlit_app.py
```
