# AI Audio Recognition

Compare two audios and gives the difference.

-----

## How it Works

The core of the project is a Python server built with the FastAPI framework. This server exposes an API endpoint that accepts two audio files as input. Here's a breakdown of the process:

1.  **Audio Loading and Normalization:** When you send two audio files to the `/compare` endpoint, the server first loads them. It converts stereo audio to mono by taking the mean of the channels and then normalizes the audio signals.

2.  **Speaker Recognition:** The server uses a pre-trained `spkrec-ecapa-voxceleb` model from SpeechBrain to extract speaker embeddings from each audio file.

3.  **Similarity Score:** It then compares these embeddings and calculates a similarity score. This score, represented as a percentage, is returned in the API response along with the processing time.

-----

## Working with the Project

### Without Docker

To run the project without Docker, you'll need to have Python and the required libraries installed on your system.

**1. Installation:**

First, install the necessary Python packages. You can do this using pip:

```bash
pip install torch torchaudio fastapi uvicorn "speechbrain==0.5.12" pydantic
```

**2. Running the Server:**

Once the dependencies are installed, you can start the server by running the `compare_audio_server.py` script:

```bash
python compare_audio_server.py
```

The server will start on `http://0.0.0.0:5000`.

**3. Sending a Request:**

You can now send a POST request to the `/compare` endpoint with two audio files. Here's an example using `curl`:

```bash
curl -X POST -F "file1=@/path/to/your/audio1.wav" -F "file2=@/path/to/your/audio2.wav" http://localhost:5000/compare
```

Replace `/path/to/your/audio1.wav` and `/path/to/your/audio2.wav` with the actual paths to your audio files.

### With Docker

Using Docker simplifies the setup process by containerizing the application and its dependencies.

**1. Build the Docker Image:**

First, you need to create a `Dockerfile` in the project's root directory.

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run compare_audio_server.py when the container launches
CMD ["uvicorn", "compare_audio_server:app", "--host", "0.0.0.0", "--port", "5000"]
```

You'll also need a `requirements.txt` file listing the dependencies:

```
torch
torchaudio
fastapi
uvicorn
speechbrain==0.5.12
pydantic
```

Now, build the Docker image with the following command:

```bash
docker build -t ai-audio-recognition .
```

**2. Run the Docker Container:**

Once the image is built, you can run it as a container:

```bash
docker run -p 5000:5000 ai-audio-recognition
```

This will start the server, and you can interact with it in the same way as the non-Docker setup, by sending requests to `http://localhost:5000/compare`.
