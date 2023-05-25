FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

#RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y libgl1-mesa-glx libpci-dev curl nano psmisc zip git vim

#RUN conda install -y faiss-gpu scikit-learn pandas flake8 yapf isort==4.3.21 yacs gdown future libgcc -c conda-forge

#RUN  apt-get --fix-broken install -y && pip install opencv-python tb-nightly matplotlib pyro-ppl==1.8.0 logger_tt tabulate tqdm wheel mccabe scipy

COPY ./fonts/* /opt/conda/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/
