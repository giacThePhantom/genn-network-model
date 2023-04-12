# This is mainly a copycat from Jamie Knight's original Dockerfile
# but aside from extending it with our image it also uses a different python version (3.10) which is shipped along with Ubuntu 22.04

ARG BASE=12.1.0-devel-ubuntu22.04
FROM nvidia/cuda:${BASE}

LABEL maintainer="Enrico.Trombetta@studenti.unitn.it"
LABEL version="0.0.1"
LABEL org.opencontainers.image.source="https://github.com/giacThePhantom/genn-network-model"
LABEL org.opencontainers.image.title="BeeGeNN Docker image"

ENV GENN_VERSION=4.8.0
ENV GENN_TARBALL=genn-${GENN_VERSION}.tar.gz

RUN apt-get update && \
    apt-get upgrade -y

# Install Python, pip and swig
RUN apt-get install -yq --no-install-recommends python3-dev python3-pip swig gosu nano wget

RUN python3 --version

# Set CUDA environment variable
ENV CUDA_PATH=/usr/local/cuda
ENV GENN_PATH=/opt/genn
ENV BEEGENN_PATH=/opt/beegenn

# setuptools version 57 does not properly support src/ layout
RUN pip install --upgrade pip setuptools

RUN wget https://github.com/genn-team/genn/archive/refs/tags/${GENN_VERSION}.tar.gz -O ${GENN_TARBALL}


RUN tar -xzf ${GENN_TARBALL}

RUN mv /genn-${GENN_VERSION} ${GENN_PATH}
# COPY genn-${GENN_VERSION} ${GENN_PATH}

WORKDIR ${GENN_PATH}

# Install GeNN and PyGeNN
RUN pip install numpy 
RUN make install -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
RUN make DYNAMIC=1 LIBRARY_DIRECTORY=${GENN_PATH}/pygenn/genn_wrapper/ -j `lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l`
RUN python3 setup.py develop

# Install Beegenn
COPY . ${BEEGENN_PATH}
WORKDIR ${BEEGENN_PATH}
RUN pip install -e .

ENTRYPOINT ["/opt/beegenn/bin/docker-start-sim.sh"]
CMD ["sim_docker"]
