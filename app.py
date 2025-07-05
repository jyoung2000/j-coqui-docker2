#!/usr/bin/env python3

"""
Coqui TTS Web Interface with Advanced Voice Cloning
A comprehensive web application for text-to-speech synthesis and voice cloning
"""

import os
import sys
import logging
import asyncio
import uuid
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import uvicorn
import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import custom modules
from modules.tts_handler import TTSHandler
from modules.voice_cloner import VoiceCloner
from modules.voice_cloner_advanced import AdvancedVoiceCloner
from modules.model_manager import ModelManager
from modules.audio_processor import AudioProcessor
from modules.database_handler import DatabaseHandler
from modules.api_handler import APIHandler
from modules.webgui import create_gradio_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Coqui TTS WebGUI",
    description="Advanced Text-to-Speech and Voice Cloning Web Interface",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger.info("Initializing TTS components...")
tts_handler = TTSHandler()
voice_cloner = VoiceCloner()
advanced_voice_cloner = AdvancedVoiceCloner()
model_manager = ModelManager()
audio_processor = AudioProcessor()
db_handler = DatabaseHandler()
api_handler = APIHandler()

# Create directories
downloads_dir = Path("/app/static/downloads")
downloads_dir.mkdir(exist_ok=True, parents=True)

# Pydantic models for API requests
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    model_name: Optional[str] = Field(None, description="TTS model name")
    speaker: Optional[str] = Field(None, description="Speaker name or ID")
    language: Optional[str] = Field("en", description="Language code")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed")
    pitch: float = Field(1.0, ge=0.5, le=2.0, description="Voice pitch")
    energy: float = Field(1.0, ge=0.5, le=2.0, description="Voice energy")
    emotion: Optional[str] = Field("neutral", description="Emotion style")
    output_format: str = Field("wav", description="Output audio format")
    stream: bool = Field(False, description="Enable streaming output")

class VoiceCloneRequest(BaseModel):
    name: str = Field(..., description="Name for the cloned voice")
    description: Optional[str] = Field(None, description="Voice description")
    speaker_wav: Optional[str] = Field(None, description="Path to speaker audio")
    quality: int = Field(7, ge=1, le=10, description="Cloning quality level")
    denoise: bool = Field(True, description="Apply denoising")

class ModelDownloadRequest(BaseModel):
    model_name: str = Field(..., description="Model name to download")
    model_type: str = Field("tts", description="Model type: tts or vocoder")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for container monitoring"""
    try:
        # Basic component checks
        models_available = len(model_manager.list_models()) > 0
        
        return {
            "status": "healthy",
            "components": {
                "tts_handler": "ok",
                "voice_cloner": "ok",
                "model_manager": "ok",
                "audio_processor": "ok",
                "models_available": models_available
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

# TTS API endpoints
@app.post("/api/tts")
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate speech from text"""
    try:
        # Validate API access
        api_handler.validate_request()
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Synthesize speech
        audio_path = await tts_handler.synthesize(
            text=request.text,
            model_name=request.model_name,
            speaker=request.speaker,
            language=request.language,
            speed=request.speed,
            pitch=request.pitch,
            energy=request.energy,
            emotion=request.emotion,
            output_format=request.output_format
        )
        
        # Create download link
        download_path = downloads_dir / f"{request_id}.{request.output_format}"
        shutil.copy(audio_path, download_path)
        
        # Log synthesis
        db_handler.log_synthesis(
            request_id=request_id,
            text=request.text,
            model_name=request.model_name,
            speaker=request.speaker,
            language=request.language
        )
        
        return {
            "success": True,
            "request_id": request_id,
            "download_url": f"/api/download/{request_id}",
            "audio_duration": tts_handler.get_duration(audio_path),
            "model_used": request.model_name or "default"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{request_id}")
async def download_audio(request_id: str):
    """Download synthesized audio file"""
    try:
        # Find file with any supported extension
        for ext in ["wav", "mp3", "ogg", "flac"]:
            file_path = downloads_dir / f"{request_id}.{ext}"
            if file_path.exists():
                return FileResponse(
                    path=str(file_path),
                    media_type=f"audio/{ext}",
                    filename=f"tts_output_{request_id}.{ext}"
                )
        
        raise HTTPException(status_code=404, detail="Audio file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@app.get("/api/models")
async def list_models():
    """List all available TTS models"""
    try:
        models = model_manager.list_models()
        return {
            "success": True,
            "models": models,
            "total_models": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/download")
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Download a new TTS model"""
    try:
        # Add download task to background
        background_tasks.add_task(
            model_manager.download_model,
            model_name=request.model_name,
            model_type=request.model_type
        )
        
        return {
            "success": True,
            "message": f"Model {request.model_name} download started",
            "model_name": request.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/{model_name}/info")
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    try:
        model_info = model_manager.get_model_info(model_name)
        return {
            "success": True,
            "model_info": model_info
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

# Voice cloning endpoints
@app.post("/api/voices/clone")
async def clone_voice(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    quality: int = Form(7),
    denoise: bool = Form(True),
    audio_file: UploadFile = File(...)
):
    """Clone a voice from uploaded audio"""
    try:
        # Save uploaded file
        temp_path = Path(f"/app/temp/{audio_file.filename}")
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Clone voice
        voice_id = await voice_cloner.clone_voice(
            audio_path=str(temp_path),
            name=name,
            description=description,
            quality=quality,
            denoise=denoise
        )
        
        # Clean up
        temp_path.unlink()
        
        return {
            "success": True,
            "voice_id": voice_id,
            "voice_name": name,
            "message": f"Voice '{name}' cloned successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voices")
async def list_voices():
    """List all cloned voices"""
    try:
        voices = voice_cloner.list_voices()
        return {
            "success": True,
            "voices": voices,
            "total_voices": len(voices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a cloned voice"""
    try:
        success = voice_cloner.delete_voice(voice_id)
        return {
            "success": success,
            "message": f"Voice {voice_id} deleted" if success else "Voice not found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voices/batch-clone")
async def batch_clone_voices(
    files: List[UploadFile] = File(...),
    base_name: str = Form(...),
    quality: int = Form(7)
):
    """Clone multiple voices in batch"""
    try:
        results = []
        for idx, file in enumerate(files):
            # Save file
            temp_path = Path(f"/app/temp/{file.filename}")
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Clone voice
            voice_name = f"{base_name}_{idx+1}"
            voice_id = await advanced_voice_cloner.clone_voice(
                audio_path=str(temp_path),
                name=voice_name,
                quality=quality
            )
            
            results.append({
                "filename": file.filename,
                "voice_id": voice_id,
                "voice_name": voice_name
            })
            
            # Clean up
            temp_path.unlink()
        
        return {
            "success": True,
            "results": results,
            "total_cloned": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_synthesis_history(limit: int = 100):
    """Get synthesis history"""
    try:
        history = db_handler.get_history(limit=limit)
        return {
            "success": True,
            "history": history,
            "total": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audio/enhance")
async def enhance_audio(
    audio_file: UploadFile = File(...),
    denoise: bool = Form(True),
    normalize: bool = Form(True),
    remove_silence: bool = Form(False)
):
    """Enhance audio quality"""
    try:
        # Save uploaded file
        temp_path = Path(f"/app/temp/{audio_file.filename}")
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Process audio
        enhanced_path = audio_processor.enhance_audio(
            audio_path=str(temp_path),
            denoise=denoise,
            normalize=normalize,
            remove_silence=remove_silence
        )
        
        # Clean up
        temp_path.unlink()
        
        # Create download link
        request_id = str(uuid.uuid4())
        download_path = downloads_dir / f"{request_id}_enhanced.wav"
        shutil.copy(enhanced_path, download_path)
        
        return {
            "success": True,
            "download_url": f"/api/download/{request_id}_enhanced",
            "original_duration": audio_processor.get_duration(temp_path),
            "enhanced_duration": audio_processor.get_duration(enhanced_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/tts-stream")
async def websocket_tts_stream(websocket):
    """WebSocket endpoint for real-time TTS streaming"""
    await websocket.accept()
    try:
        while True:
            # Receive text
            data = await websocket.receive_json()
            
            # Stream TTS
            async for chunk in tts_handler.stream_synthesis(
                text=data.get("text", ""),
                model_name=data.get("model_name"),
                speaker=data.get("speaker"),
                language=data.get("language", "en")
            ):
                await websocket.send_bytes(chunk)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

# Create and mount Gradio interface
gradio_app = create_gradio_interface(
    tts_handler=tts_handler,
    voice_cloner=voice_cloner,
    advanced_voice_cloner=advanced_voice_cloner,
    model_manager=model_manager,
    audio_processor=audio_processor
)

# Mount Gradio app at root
app = gr.mount_gradio_app(app, gradio_app, path="/")

# API documentation
@app.get("/api/docs")
async def api_documentation():
    """Get API documentation"""
    return {
        "endpoints": {
            "tts": {
                "POST /api/tts": "Generate speech from text",
                "GET /api/download/{request_id}": "Download synthesized audio"
            },
            "models": {
                "GET /api/models": "List all available models",
                "POST /api/models/download": "Download a new model",
                "GET /api/models/{model_name}/info": "Get model information"
            },
            "voices": {
                "POST /api/voices/clone": "Clone a voice from audio",
                "GET /api/voices": "List all cloned voices",
                "DELETE /api/voices/{voice_id}": "Delete a cloned voice",
                "POST /api/voices/batch-clone": "Clone multiple voices"
            },
            "audio": {
                "POST /api/audio/enhance": "Enhance audio quality"
            },
            "history": {
                "GET /api/history": "Get synthesis history"
            },
            "websocket": {
                "WS /ws/tts-stream": "Real-time TTS streaming"
            }
        }
    }

# Main entry point
if __name__ == "__main__":
    # Get ports from environment - use 8080 for cloud deployment
    web_port = int(os.getenv("PORT", os.getenv("WEB_PORT", "8080")))
    
    # Create necessary directories
    for dir_name in ["models", "output", "voice_samples", "user_data", "config", "logs", "temp"]:
        Path(f"/app/{dir_name}").mkdir(exist_ok=True, parents=True)
    
    # Initialize database
    db_handler.initialize()
    
    # Load default models
    logger.info("Loading default models...")
    model_manager.load_default_models()
    
    # Run the server
    logger.info(f"Starting Coqui TTS WebGUI on port {web_port}")
    logger.info(f"Web UI: http://localhost:{web_port}")
    logger.info(f"API Docs: http://localhost:{web_port}/api/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=web_port,
        reload=False,
        log_level="info"
    )