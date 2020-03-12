#ARG IMAGE_REGISTRY=gitlab.kitware.com:4567/opengeoscience/viameweb
#ARG BASE_IMAGE
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 as base

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update -qq && apt-get install -yq \
        curl \
        apt-transport-https \
        ca-certificates \
        software-properties-common \
        python3-software-properties \
    && apt-get install -qq --no-install-recommends \
        build-essential \
        git \
        ninja-build \
        pkg-config \
        python3 \
        ssh \
        unzip \
        wget \
        r-base \
        libffi-dev \
        libssl-dev \
        libjpeg-dev \
        zlib1g-dev \
        libpython3-dev \
	python3-distutils \
    && rm -rf /var/lib/apt/lists/*


## Install latest version of Cmake
RUN curl https://apt.kitware.com/keys/kitware-archive-latest.asc -L 2>/dev/null | apt-key add - \
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && apt-get update -y \
    && apt-get install -y \
        cmake \
    && rm -rf /var/lib/apt/lists/*


RUN curl https://bootstrap.pypa.io/get-pip.py -o /root/get-pip.py \
    && python3 /root/get-pip.py

FROM base as build

RUN git clone https://github.com/girder/girder_worker.git /girder_worker/
WORKDIR /girder_worker

RUN rm -rf ./dist && python3 setup.py sdist


FROM base
COPY --from=build /girder_worker/dist/*.tar.gz /
COPY --from=build /girder_worker/docker-entrypoint.sh /docker-entrypoint.sh
RUN pip3 install /*.tar.gz

RUN useradd -D --shell=/bin/bash && useradd -m worker

RUN chown -R worker:worker /usr/local/lib/python3.6/dist-packages/girder_worker

USER worker

RUN girder-worker-config set celery broker "amqp://%(RABBITMQ_USER)s:%(RABBITMQ_PASS)s@%(RABBITMQ_HOST)s/"


VOLUME /girder_worker

ENV PYTHON_BIN=python3

ENTRYPOINT ["/docker-entrypoint.sh"]
