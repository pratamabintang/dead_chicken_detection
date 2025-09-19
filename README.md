# Deep Learning Approach for Detection of Dead Chicken Based on Visible and Thermal Images
This repository presents a deep learning-based approach for automatic detection of dead chickens using both visible and thermal images. The method leverages computer vision techniques to enhance accuracy in smart poultry farming, supporting early detection and efficient farm management

---

## Installation
Clone this repository:
```
https://github.com/pratamabintang/dead_chicken_detection.git
cd dead_chicken_detection
```

## Setup Depedency
```
python -m venv development
development\Scripts\activate.bat
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
```
Please adjust the CUDA version in the requirements.txt according to your GPU. You may need a different version, and remember to update the index URL accordingly

## Train, Inference, Performance
```
#Train
python fasterRCNN.py

#Inference
python inference.py

#Training Performance Check
python performance.py
```
Please adjust every parameter and data path in the code file. I have added comments to help you align it with your dataset path
