# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Git LFS
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Set build argument for Hugging Face token
ARG HF_AUTH_TOKEN
ENV HF_AUTH_TOKEN=${HF_AUTH_TOKEN}

# Clone the Hugging Face model repository
RUN bash -c 'git clone https://${HF_AUTH_TOKEN}@huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct /usr/src/app/Meta-Llama-3-70B-Instruct'

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
