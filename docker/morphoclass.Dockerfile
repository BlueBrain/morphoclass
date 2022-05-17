FROM nvidia/cuda:10.2-runtime

ENV http_proxy="http://bbpproxy.epfl.ch:80"
ENV https_proxy="http://bbpproxy.epfl.ch:80"
ENV HTTP_PROXY="http://bbpproxy.epfl.ch:80"
ENV HTTPS_PROXY="http://bbpproxy.epfl.ch:80"

# Debian's default LANG=C breaks python3.
# See commends in the official python docker file:
# https://github.com/docker-library/python/blob/master/3.7/buster/Dockerfile
ENV LANG=C.UTF-8
ENV TZ="Europe/Zurich"

# CUDA Linux Repo Key Rotation: https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

#Install system packages
RUN \
apt-get update && \
DEBIAN_FRONTEND="noninteractive" \
apt-get install -y --no-install-recommends \
    build-essential make curl git htop less man ssh tzdata vim wget

# Install python
RUN \
DEBIAN_FRONTEND="noninteractive" \
apt-get install -y --no-install-recommends \
python3.8-dev python3.8-venv python3-pip && \
python3.8 -m pip install --upgrade pip setuptools wheel && \
update-alternatives --install /usr/local/bin/python python /usr/bin/python3.8 0

# Install requirements
COPY requirements*.txt /tmp/
RUN \
pip install -r /tmp/requirements.txt && \
pip install -r /tmp/requirements-extras.txt

# Install torch geometric
RUN \
TORCH=$(pip freeze | grep torch== | sed -re "s/torch==([^+]+).*/\1/") && \
CUDA=cu102 && \
FIND_LINKS="https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html" && \
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv --no-index -f $FIND_LINKS && \
pip install -q "torch-geometric<2"

# Install the latest jupyter lab and ipywidgets
RUN pip install -U jupyterlab ipywidgets

# Add and configure users
SHELL ["/bin/bash", "-c"]
ARG MORPHOCLASS_USERS
RUN echo Custom users: $MORPHOCLASS_USERS
COPY docker/utils.sh /tmp
RUN \
. /tmp/utils.sh && \
groupadd -g 999 docker && \
create_users "${MORPHOCLASS_USERS},guest/1000" "docker" && \
configure_user

# Entry point
EXPOSE 8888
RUN mkdir /workdir && chmod a+rwX /workdir
WORKDIR /workdir
USER guest
ENTRYPOINT ["env"]
CMD ["bash", "-l"]
