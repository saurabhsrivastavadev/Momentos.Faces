# Let's base off python ubuntu image 
FROM python:3.10.7-bullseye

# Install python libraries
RUN pip install mediapipe
RUN pip install cmake
RUN pip install dlib
RUN pip install face_recognition

# opencv libraries for fixing below error 
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

# Copy the app code
COPY ./app/. /app/.
