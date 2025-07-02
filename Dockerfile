# Use an official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    libportaudio2 \
    libasound2-dev \
    ffmpeg \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in separate steps
RUN pip install numpy

# Install torch and torchaudio separately (large packages)
RUN pip install torch torchaudio

# Install other dependencies
RUN pip install \
    soundfile \
    librosa \
    fastapi \
    uvicorn \
    python-multipart

# Install sox after numpy is available
RUN pip install sox

# Install SpeechBrain
RUN pip install git+https://github.com/speechbrain/speechbrain.git@develop

# Create necessary directories
RUN mkdir -p /app/audio /app/pretrained_models

# Copy the server script
COPY scripts/compare_audio_server.py /app/

# Create initialization script
RUN echo 'from speechbrain.inference.speaker import SpeakerRecognition\nverification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")' > /app/init_models.py

# Create entrypoint script
RUN echo '#!/bin/bash\nif [ ! -d "/app/pretrained_models/spkrec-ecapa-voxceleb" ]; then\n    python /app/init_models.py || exit 1\nfi\nexec uvicorn compare_audio_server:app --host 0.0.0.0 --port 5000' > /app/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose the port
EXPOSE 5000

# Set entry point
ENTRYPOINT ["/app/entrypoint.sh"]