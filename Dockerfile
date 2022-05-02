FROM tensorflow/tensorflow:latest

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ENV DEBIAN_FRONTEND noninteractive
RUN DEBIAN_FRONTEND=noninteractive  apt-get update && apt-get install -y --no-install-recommends \
            build-essential  \
            python3          \
            python3-pip      \  
            python3-opencv     
ENV DEBIAN_FRONTEND noninteractive

RUN pip3 install --no-cache-dir \
    pip                         \ 
    setuptools

RUN python3 -m pip install --upgrade pip

RUN pip3 --no-cache-dir install --upgrade \    
    tensorflow_hub \
    numpy          \
    opencv-python  \
    pillow         \
    scipy          \
    termcolor      \
    alive-progress


