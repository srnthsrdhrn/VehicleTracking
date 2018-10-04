FROM tensorflow/tensorflow:latest-gpu-py3
RUN apt-get update
RUN mkdir /usr/srinath
WORKDIR /usr/srinath
RUN pip install cython
COPY darkflow/ /usr/srinath/darkflow
WORKDIR /usr/srinath/darkflow
RUN python setup.py build_ext --inplace
RUN pip install .
ENV CFLAGS="-I /usr/local/lib/python3.5/dist-packages/numpy/core/include $CFLAGS"
ENV SERVER=true
RUN pip3 install opencv-python
RUN apt-get install -y libxrender-dev libsm6 libxext6 python3-tk
COPY requirements.txt /usr/srinath/
WORKDIR /usr/srinath
RUN pip install -r requirements.txt
RUN mkdir media
RUN mkdir media/tmp
RUN mkdir /usr/srinath/media/output_file
COPY . /usr/srinath