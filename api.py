from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import soundfile as sf
import tempfile
import torch
import os
from pathlib import Path

from vieneutts import VieNeuTTS

# Initialize FastAPI app
app = FastAPI(
    title="VieNeu-TTS API",
    description="Vietnamese Text-to-Speech API with voice cloning",
    version="1.0.0"
)

# Initialize TTS model
print("⏳ Initializing VieNeu-TTS...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device.upper()}")

tts = VieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS",
    backbone_device=device,
    codec_repo="neuphonic/neucodec",
    codec_device=device
)
print("✅ Model loaded successfully!")

# Voice samples configuration
VOICE_SAMPLES = {
    "nam_1": {
        "audio": "./sample/id_0001.wav",
        "text": "./sample/id_0001.txt",
        "description": "Nam 1 - Giọng nam miền Nam"
    },
    "nu_1": {
        "audio": "./sample/id_0002.wav",
        "text": "./sample/id_0002.txt",
        "description": "Nữ 1 - Giọng nữ miền Nam"
    },
    "nam_2": {
        "audio": "./sample/id_0003.wav",
        "text": "./sample/id_0003.txt",
        "description": "Nam 2 - Giọng nam miền Nam"
    },
    "nu_2": {
        "audio": "./sample/id_0004.wav",
        "text": "./sample/id_0004.txt",
        "description": "Nữ 2 - Giọng nữ miền Nam"
    },
    "nam_3": {
        "audio": "./sample/id_0005.wav",
        "text": "./sample/id_0005.txt",
        "description": "Nam 3 - Giọng nam miền Nam"
    },
    "nam_4": {
        "audio": "./sample/id_0007.wav",
        "text": "./sample/id_0007.txt",
        "description": "Nam 4 - Giọng nam miền Nam"
    }
}

# Request models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (max 500 characters)", max_length=500)
    voice_id: str = Field(..., description="Voice sample ID (nam_1, nu_1, nam_2, nu_2, nam_3, nam_4)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Xin chào, đây là giọng nói được tổng hợp từ VieNeu-TTS.",
                "voice_id": "nam_1"
            }
        }

class VoiceInfo(BaseModel):
    voice_id: str
    description: str

# Endpoints
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "VieNeu-TTS API",
        "version": "1.0.0",
        "endpoints": {
            "voices": "/voices",
            "synthesize": "/tts",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": device,
        "model": "VieNeu-TTS"
    }

@app.get("/voices", response_model=list[VoiceInfo])
async def list_voices():
    """Get list of available voice samples"""
    return [
        {"voice_id": voice_id, "description": info["description"]}
        for voice_id, info in VOICE_SAMPLES.items()
    ]

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Synthesize speech from text using selected voice sample
    
    - **text**: Input text to synthesize (Vietnamese)
    - **voice_id**: Voice sample ID (nam_1, nu_1, nam_2, nu_2, nam_3, nam_4)
    
    Returns: WAV audio file
    """
    try:
        # Validate text
        if not request.text or request.text.strip() == "":
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 500:
            raise HTTPException(status_code=400, detail="Text too long (max 500 characters)")
        
        # Validate voice_id
        if request.voice_id not in VOICE_SAMPLES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice_id. Available voices: {', '.join(VOICE_SAMPLES.keys())}"
            )
        
        # Get reference audio and text
        voice_info = VOICE_SAMPLES[request.voice_id]
        ref_audio_path = voice_info["audio"]
        ref_text_path = voice_info["text"]
        
        # Check if files exist
        if not os.path.exists(ref_audio_path):
            raise HTTPException(status_code=500, detail=f"Reference audio not found: {ref_audio_path}")
        if not os.path.exists(ref_text_path):
            raise HTTPException(status_code=500, detail=f"Reference text not found: {ref_text_path}")
        
        # Read reference text
        with open(ref_text_path, "r", encoding="utf-8") as f:
            ref_text = f.read()
        
        print(f"🎤 Processing request with voice: {request.voice_id}")
        print(f"📝 Text: {request.text[:50]}...")
        
        # Encode reference audio
        ref_codes = tts.encode_reference(ref_audio_path)
        
        # Synthesize speech
        print(f"🎵 Synthesizing speech on {device.upper()}...")
        wav = tts.infer(request.text, ref_codes, ref_text)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, wav, 24000)
            output_path = tmp_file.name
        
        print("✅ Synthesis completed!")
        
        # Return audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"tts_{request.voice_id}.wav",
            headers={
                "X-Voice-ID": request.voice_id,
                "X-Text-Length": str(len(request.text))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)