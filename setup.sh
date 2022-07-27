#!/bin/bash

sudo apt-get install -y \
    libopencv-core-dev \
    libopencv-highgui-dev \
    libopencv-calib3d-dev \
    libopencv-features2d-dev \
    libopencv-imgproc-dev \
    libopencv-video-dev \
    build-essential cmake \
    libx11-dev libgtk-3-dev \
    python3.9 python3.9-dev python3.9-tk \
    qt4-dev-tools libatlas-base-dev libhdf5-103

pip install -r requirements.txt

pip install tensorflow-object-detection-api