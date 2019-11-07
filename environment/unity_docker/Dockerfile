# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV LANG C.UTF-8

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    libexpat1 \
    libffi6 \
    libgdbm3 \
    libreadline6 \
    libsqlite3-0 \
    libssl1.0.0 \
    xserver-xorg-dev \
    gconf-service \
    lib32gcc1 \
    lib32stdc++6 \
    libasound2 \
    libarchive13 \
    libc6 \
    libc6-i386 \
    libcairo2 \
    libcap2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libfreetype6 \
    libgcc1 \
    libgconf-2-4 \
    libgdk-pixbuf2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglu1-mesa \
    libgtk2.0-0 \
    libgtk3.0 \
    libnotify4 \
    libnspr4 \
    libnss3 \
    libpango1.0-0 \
    libsoup2.4-1 \
    libstdc++6 \
    libx11-6 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libunwind-dev \
    zlib1g \
    pulseaudio \
    debconf \
    npm \
    xdg-utils \
    lsb-release \
    libpq5 \
    xvfb \
    wget \
    ffmpeg \
    libglu1-mesa-dev \
    freeglut3-dev \
    mesa-common-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
RUN virtualenv --python=python3.6 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements.txt /mujoco_py/
COPY ./requirements.dev.txt /mujoco_py/
RUN pip install -r requirements.txt
RUN pip install -r requirements.dev.txt
RUN pip install cloudpickle==1.2.1
RUN pip install cached-property==1.3.1
RUN pip install mujoco_py==1.50.1.59
# For atari-py
RUN apt-get update && apt-get install -y zlib1g-dev swig cmake
RUN pip install gym[all]==0.10.5
RUN pip install gitpython==2.1.7
RUN pip install gtimer==1.0.0b5
RUN pip install pygame
RUN pip install awscli==1.11.179
RUN pip install boto3==1.4.8
RUN pip install dominate==2.3.1
RUN pip install path.py==10.3.1

RUN pip install http://download.pytorch.org/whl/cu90/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install joblib==0.9.4

# If you're curious what the DEBIAN_FRONTEND is all about:
# https://askubuntu.com/questions/506158/unable-to-initialize-frontend-dialog-when-using-ssh
#RUN apt-get purge nvidia*
#RUN add-apt-repository --yes ppa:graphics-drivers && apt-get update

RUN apt-get update && apt-get install -y libav-tools

# For some reason this works despite an error showing up...
# RUN DEBIAN_FRONTEND=noninteractive apt-get -qy install nvidia-384; exit 0
# ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-384

RUN mkdir /root/code
WORKDIR /root/code

# trigger the mujoco_py cythonization
RUN python -c 'import mujoco_py'

RUN pip install opencv-python==3.4.0.12
RUN pip install torchvision==0.2.0
RUN pip install sk-video==1.1.10
RUN pip install matplotlib==2.2.2
RUN pip install numpy-stl==2.7.0
RUN pip install pyquaternion==0.9.2
RUN pip install moviepy==0.2.3.5
RUN pip install scikit-image
RUN pip install cryptography
RUN pip install google-api-python-client==1.7.8
RUN pip install gsutil
RUN pip install psutil
RUN pip install h5py
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y git wget cmake psmisc
RUN apt-get install -y xvfb
# Figure out how to get ray to run without this
ENV PATH /env/bin:${PATH}
ENV LD_LIBRARY_PATH /home/ubuntu/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
RUN pip install pyqt5


RUN wget -nv https://beta.unity3d.com/download/5f859a4cfee5/UnitySetup-2019.2.11f1 -O UnitySetup && \
    # compare sha1 if given
    if [ -n "${SHA1}" -a "${SHA1}" != "" ]; then \
      echo "${SHA1}  UnitySetup" | sha1sum --check -; \
    else \
      echo "no sha1 given, skipping checksum"; \
    fi && \
    # make executable
    chmod +x UnitySetup && \
    # agree with license
    echo y | \
    # install unity with required components
    xvfb-run --auto-servernum --server-args='-screen 0 640x480x24' \
    ./UnitySetup \
    --unattended \
    --install-location=/opt/Unity \
    --verbose \
    --download-location=/tmp/unity \
    --components=Unity && \
    # remove setup & temp files
    rm UnitySetup && \
    rm -rf /tmp/unity && \
    rm -rf /root/.local/share/Trash/*

RUN mkdir -p /root/.local/share/unity3d/Certificates/ && \
    mkdir -p /root/.local/share/unity3d/Unity/ && \
    /opt/Unity/Editor/Unity -batchmode -quit -nographics -createManualActivationFile -logfile /dev/stdout || :

ADD conf/CACerts.pem /root/.local/share/unity3d/Certificates/
ADD conf/asound.conf /etc/

RUN git clone https://github.com/jcoreyes/ml-agents
WORKDIR ml-agents/ml-agents-envs
RUN pip install -e ./

RUN cd ../ml-agents && pip install -e ./

WORKDIR /root/code
EXPOSE 5005