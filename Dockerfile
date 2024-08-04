# # Base & applications
# FROM python:3.7-slim
# RUN apt-get update && apt-get install -y libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx vim git

# # Define workdir (ref: https://docs.docker.com/engine/reference/builder/#workdir)
# WORKDIR /home/ubuntu/mad/


# # Mlaaslib
# ARG PIP_INDEX_URL
# RUN pip install --index-url $PIP_INDEX_URL --trusted-host pi.madstreetden.xyz mlaaslib==1.0

# COPY tag-product2dotx/ /home/ubuntu/mad/llmnode/

# # Python packages
# RUN pip install --index-url https://pypi.org/simple/ -r /home/ubuntu/mad/llmnode/requirements.txt
# RUN pip install azure-storage-blob azure-identity azure-keyvault-secrets

# COPY tag-product2dotx/hpt_setup.sh /home/ubuntu/mad/llmnode
# RUN chmod +x /home/ubuntu/mad/llmnode/hpt_setup.sh
# RUN /home/ubuntu/mad/llmnode/hpt_setup.sh
# ENV PYTHONPATH="/home/ubuntu/mad/:/home/ubuntu/"

# ENTRYPOINT ["python", "/home/ubuntu/mad/llmnode/side_car.py"]



FROM python:3.8-slim
ARG PIP_INDEX_URL

# User creation
RUN useradd ubuntu
USER root

#nvidia-cuda base
ENV NVARCH x86_64

ENV NVIDIA_REQUIRE_CUDA "cuda>=11.6 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471"
ENV NV_CUDA_CUDART_VERSION 11.6.55-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-11-6

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
RUN apt-get update
RUN apt-get install -y wget
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 11.6.0

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-6=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-11.6 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Add system requirements
RUN apt update
RUN pip install ipython
RUN apt install elpa-magit -y
RUN apt install openssh-server -y
RUN apt install git-all -y
RUN apt install libglib2.0-0 libsm6 libxrender1 vim -y

# make a working directory
RUN mkdir -p /home/ubuntu/mad/

# copy req.txt and install
COPY tag-product2dotx/requirements.txt /home/ubuntu/mad/requirements.txt

# DO NOT CHANGE MLAASLIB VERSION for GPU docker
RUN pip install --index-url $PIP_INDEX_URL --trusted-host pi.madstreetden.xyz mlaaslib==2.0.0

RUN pip install --index-url https://pypi.org/simple/ -r /home/ubuntu/mad/requirements.txt --no-cache-dir
RUN pip install --index-url https://pypi.org/simple/ azure-storage-blob

# copy entire repo

# set env
ENV PYTHONPATH="$PYTHONPATH:/home/ubuntu/mad/llmnode/"

ENTRYPOINT ["python", "/home/ubuntu/mad/llmnode/side_car.py"]
