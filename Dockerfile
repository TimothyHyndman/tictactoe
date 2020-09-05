FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

#RUN apt-get update && apt-get install -y \
#    libsdl1.2-dev \
#    libsdl-image1.2-dev \
#    libsdl-mixer1.2-dev \
#    libsdl-ttf2.0-dev \
#    libportmidi-dev

COPY requirements_training.txt .
RUN pip install -r requirements_training.txt