#FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

#RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get install qemu-arm-static
RUN apt-get update 
RUN apt-get install -y libgl1-mesa-glx 
RUN apt-get install -y libpci-dev 
RUN apt-get install -y curl 
RUN apt-get install -y nano 
RUN apt-get install -y psmisc 
RUN apt-get install -y zip 
RUN apt-get install -y git
RUN apt-get install -y vim 
RUN apt-get install -y iputils-ping

# CONDA
#RUN conda create -n rice python=3 -y
#RUN conda activate rice
#RUN echo "conda activate rice" >> ~/.bashrc
#SHELL ["/bin/bash", "--login", "-c"]

RUN conda install -y faiss-gpu scikit-learn pandas flake8 yapf isort yacs gdown future libgcc -c conda-forge

# PIP
#RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN  apt-get --fix-broken install -y && pip install opencv-python tb-nightly matplotlib logger_tt tabulate tqdm wheel mccabe scipy

COPY ./fonts/* /opt/conda/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/
