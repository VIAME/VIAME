FROM gitlab.kitware.com:4567/opengeoscience/viameweb/base/girder_worker

USER root

RUN apt-get update -y &&\
    apt-get install -y \
    apt-transport-https \
    ca-certificates \
    git \
    openssl \
    software-properties-common \
    wget \
    zip unzip \
    libgl1-mesa-dev \
    libexpat1-dev \
    libgtk2.0-dev \
    libxt-dev \
    libxml2-dev \
    libssl-dev \
    liblapack-dev \
    python3-dev \
    zlib1g-dev \
    ffmpeg

RUN git clone https://github.com/Kitware/VIAME.git /home/VIAME &&\
    cd /home/VIAME && git submodule update --init --recursive

# Need CMake >= 3.11.4 for VIAME,
# Thus, install latest stable CMake 3.14.1
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - &&\
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' &&\
    apt-get update -qq &&\
    apt-get install -qq cmake &&\
    apt-get install -qq --no-install-recommends \
        build-essential\
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN cd /home/VIAME && mkdir build && cd build &&\
    cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DVIAME_ENABLE_CUDA=OFF \
    -DVIAME_ENABLE_CUDNN=OFF \
    -DVIAME_ENABLE_SMQTK=OFF \
    -DVIAME_ENABLE_VIVIA=OFF \
    -DVIAME_ENABLE_KWANT=OFF \
    -DVIAME_ENABLE_PYTHON=OFF \
    -DVIAME_ENABLE_PYTORCH=OFF \
    -DVIAME_ENABLE_DARKNET=ON \
    .. && make -j8

WORKDIR /home

COPY docker/provision provision

COPY server viame_girder

RUN cd viame_girder && pip install --no-cache-dir -e .

USER worker

CMD girder-worker -l info
