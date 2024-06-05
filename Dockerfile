# Use the official Python 3.9 image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install any dependencies specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose a port (if your application listens on a specific port)
# EXPOSE 8080

# Specify the command to run when the container starts
CMD ["python", "cnn_lstm.py"]
