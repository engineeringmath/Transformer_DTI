FROM tensorflow/tensorflow:1.14.0-gpu-py3
# FROM python:3.6-stretch
# RUN apt-get update && apt-get upgrade -y

MAINTAINER Davood Karimi <davood.karimi@gmail.com>

# install build utilities
# RUN apt-get update 
#RUN apt-get upgrade -y

# set the working directory for containers
WORKDIR  /src/transfoermer_dti

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY icc.py  /src/

# Copy all the files from the project’s root to the working directory
COPY .   /src/

# Running Python Application
CMD ["python3", "/src/DTI_transformer.py"]

