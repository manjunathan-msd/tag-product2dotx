# Base & applications
FROM python:3.7-slim
RUN apt-get update && apt-get install -y libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-glx vim git

# Define workdir (ref: https://docs.docker.com/engine/reference/builder/#workdir)
WORKDIR /home/ubuntu/mad/


# Mlaaslib
ARG PIP_INDEX_URL
RUN pip install --index-url $PIP_INDEX_URL --trusted-host pi.madstreetden.xyz mlaaslib==1.0

COPY tag-product2dotx/ /home/ubuntu/mad/llmnode/

# Python packages
RUN pip install --index-url https://pypi.org/simple/ -r /home/ubuntu/mad/llmnode/requirements.txt
RUN pip install azure-storage-blob azure-identity azure-keyvault-secrets

COPY tag-product2dotx/hpt_setup.sh /home/ubuntu/mad/llmnode
RUN chmod +x /home/ubuntu/mad/llmnode/hpt_setup.sh
RUN /home/ubuntu/mad/llmnode/hpt_setup.sh
ENV PYTHONPATH="/home/ubuntu/mad/:/home/ubuntu/"

ENTRYPOINT ["python", "/home/ubuntu/mad/llmnode/address_extraction/side-car.py"]
