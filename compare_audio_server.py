import os
import torch
import torchaudio
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.dataio.preprocess import AudioNormalizer
import tempfile

app = FastAPI(
    title="Audio Comparison API",
    description="API for comparing two audio files and calculating their similarity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model globally
print("Loading model...")
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
audio_normalizer = AudioNormalizer()

def load_and_normalize(file_path):
    signal, sr = torchaudio.load(file_path)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    signal = signal.transpose(0, 1)
    signal = signal.to(verification.device)
    signal = audio_normalizer(signal, sr)
    return signal

class ComparisonResponse(BaseModel):
    similarity: float
    processing_time: float

@app.post("/compare", response_model=ComparisonResponse)
async def compare_audio(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    start_time = time.time()
    try:
        # Save temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp1, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp2:
            
            # Save uploaded files
            content1 = await file1.read()
            content2 = await file2.read()
            temp1.write(content1)
            temp2.write(content2)
            temp1.flush()
            temp2.flush()

            try:
                # Process audio files
                waveform1 = load_and_normalize(temp1.name)
                waveform2 = load_and_normalize(temp2.name)

                # Add batch dimension
                batch1 = waveform1.unsqueeze(0)
                batch2 = waveform2.unsqueeze(0)

                # Verify
                score, decision = verification.verify_batch(batch1, batch2)
                similarity_score = score.item() * 100

                total_time = time.time() - start_time
                
                return ComparisonResponse(
                    similarity=similarity_score,
                    processing_time=total_time
                )

            finally:
                # Clean up temp files
                os.unlink(temp1.name)
                os.unlink(temp2.name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000) 