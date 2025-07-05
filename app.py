import os
import sys
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

import gradio as gr
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import torch
from TTS.api import TTS

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from modules.tts_handler import TTSHandler
from modules.voice_cloner import VoiceCloner
from modules.voice_cloner_advanced import AdvancedVoiceCloner
from modules.model_manager import ModelManager
from modules.audio_processor import AudioProcessor
from modules.webgui import create_gradio_interface
from modules.api_handler import APIHandler
from modules.database_handler import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Coqui TTS WebGUI",
    version="2.0.0",
    description="Advanced TTS Web Interface with Voice Cloning and Comprehensive API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize handlers
tts_handler = TTSHandler()
voice_cloner = VoiceCloner()
advanced_voice_cloner = AdvancedVoiceCloner()
model_manager = ModelManager()
audio_processor = AudioProcessor()
api_handler = APIHandler()
db_handler = DatabaseHandler()

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Create downloads directory
downloads_dir = Path(__file__).parent / "static" / "downloads"
downloads_dir.mkdir(exist_ok=True, parents=True)

# Pydantic models for API
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "coqui-tts-webgui",
        "version": "2.0.0",
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": len(model_manager.loaded_models)
    }

# API routes
@app.post("/api/tts")
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate speech from text with advanced options"""
    try:
        # Generate unique ID for this request
        request_id = str(uuid.uuid4())
        
        # Log request
        logger.info(f"TTS request {request_id}: {request.text[:50]}...")
        
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
            output_format=request.output_format,
            stream=request.stream
        )
        
        # Save to database
        db_handler.save_synthesis(
            request_id=request_id,
            text=request.text,
            audio_path=str(audio_path),
            settings=request.dict()
        )
        
        # Create download link
        download_path = downloads_dir / f"{request_id}.{request.output_format}"
        shutil.copy(audio_path, download_path)
        
        return {
            "success": True,
            "request_id": request_id,
            "audio_path": str(audio_path),
            "download_url": f"/api/download/{request_id}",
            "duration": audio_processor.get_duration(audio_path)
        }
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{request_id}")
async def download_audio(request_id: str):
    """Download synthesized audio"""
    try:
        # Find the audio file
        files = list(downloads_dir.glob(f"{request_id}.*"))
        if not files:
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        return FileResponse(
            files[0],
            media_type="audio/wav",
            filename=f"tts_{request_id}.{files[0].suffix}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List all available TTS models with details"""
    try:
        models = model_manager.list_all_models()
        return {
            "success": True,
            "models": models,
            "total": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/download")
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Download a new TTS model"""
    try:
        # Start download in background
        background_tasks.add_task(
            model_manager.download_model,
            request.model_name,
            request.model_type
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
        raise HTTPException(status_code=500, detail=str(e))

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
        upload_dir = Path("/app/temp/uploads")
        upload_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}.wav"
        with open(file_path, "wb") as f:
            f.write(await audio_file.read())
        
        # Clone voice
        voice_id = await voice_cloner.clone_voice(
            name=name,
            description=description,
            audio_path=str(file_path),
            quality=quality,
            denoise=denoise
        )
        
        # Clean up temp file
        file_path.unlink()
        
        return {
            "success": True,
            "voice_id": voice_id,
            "name": name,
            "message": "Voice cloned successfully"
        }
    except Exception as e:
        logger.error(f"Voice cloning error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voices")
async def list_voices():
    """List all available voices"""
    try:
        voices = voice_cloner.list_voices()
        return {
            "success": True,
            "voices": voices,
            "total": len(voices)
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
            "message": "Voice deleted successfully" if success else "Voice not found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voices/batch-clone")
async def batch_clone_voices(
    files: List[UploadFile] = File(...),
    base_name: str = Form(...),
    quality: int = Form(7)
):
    """Clone multiple voices from uploaded audio files"""
    try:
        results = []
        upload_dir = Path("/app/temp/uploads")
        upload_dir.mkdir(exist_ok=True, parents=True)
        
        for i, file in enumerate(files):
            # Save uploaded file
            file_path = upload_dir / f"{uuid.uuid4()}.wav"
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            # Clone voice
            voice_name = f"{base_name}_{i+1}"
            try:
                voice_id = await voice_cloner.clone_voice(
                    name=voice_name,
                    description=f"Batch cloned voice {i+1}",
                    audio_path=str(file_path),
                    quality=quality
                )
                results.append({
                    "success": True,
                    "voice_id": voice_id,
                    "name": voice_name,
                    "original_filename": file.filename
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "original_filename": file.filename
                })
            finally:
                # Clean up temp file
                file_path.unlink()
        
        return {
            "success": True,
            "results": results,
            "total_processed": len(files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_synthesis_history(limit: int = 100):
    """Get synthesis history"""
    try:
        history = db_handler.get_synthesis_history(limit)
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
    """Enhance uploaded audio"""
    try:
        # Save uploaded file
        upload_dir = Path("/app/temp/uploads")
        upload_dir.mkdir(exist_ok=True, parents=True)
        
        file_path = upload_dir / f"{uuid.uuid4()}.wav"
        with open(file_path, "wb") as f:
            f.write(await audio_file.read())
        
        # Enhance audio
        enhanced_path = audio_processor.enhance_audio(
            audio_path=str(file_path),
            denoise=denoise,
            normalize=normalize,
            remove_silence=remove_silence
        )
        
        # Create download link
        download_id = str(uuid.uuid4())
        download_path = downloads_dir / f"{download_id}.wav"
        shutil.copy(enhanced_path, download_path)
        
        # Clean up temp files
        file_path.unlink()
        if enhanced_path != str(file_path):
            Path(enhanced_path).unlink()
        
        return {
            "success": True,
            "download_url": f"/api/download/{download_id}",
            "original_filename": audio_file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/tts-stream")
async def websocket_tts_stream(websocket):
    """WebSocket endpoint for streaming TTS"""
    await websocket.accept()
    try:
        while True:
            # Receive text from client
            data = await websocket.receive_json()
            text = data.get("text", "")
            
            if not text:
                continue
            
            # Generate speech
            audio_path = await tts_handler.synthesize(
                text=text,
                stream=True
            )
            
            # Read audio file and send chunks
            with open(audio_path, "rb") as f:
                while True:
                    chunk = f.read(1024)
                    if not chunk:
                        break
                    await websocket.send_bytes(chunk)
            
            # Send end marker
            await websocket.send_json({"status": "complete"})
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.get("/api/docs")
async def api_documentation():
    """Get API documentation"""
    return {
        "title": "Coqui TTS WebGUI API",
        "version": "2.0.0",
        "description": "Advanced TTS Web Interface with Voice Cloning",
        "endpoints": {
            "POST /api/tts": "Generate speech from text",
            "GET /api/models": "List available TTS models",
            "POST /api/models/download": "Download new TTS model",
            "GET /api/models/{model_name}/info": "Get model information",
            "POST /api/voices/clone": "Clone voice from audio",
            "GET /api/voices": "List cloned voices",
            "DELETE /api/voices/{voice_id}": "Delete cloned voice",
            "POST /api/voices/batch-clone": "Clone multiple voices",
            "GET /api/history": "Get synthesis history",
            "POST /api/audio/enhance": "Enhance audio quality",
            "GET /api/download/{request_id}": "Download generated audio",
            "WS /ws/tts-stream": "Streaming TTS WebSocket"
        }
    }

def create_gradio_app():
    """Create Gradio interface"""
    return create_gradio_interface(
        tts_handler=tts_handler,
        voice_cloner=voice_cloner,
        model_manager=model_manager,
        audio_processor=audio_processor
    )

async def startup_event():
    """Application startup event"""
    logger.info("Starting Coqui TTS WebGUI...")
    
    # Initialize database
    db_handler.initialize()
    
    # Load default models
    await model_manager.load_default_models()
    
    logger.info("Application started successfully")

if __name__ == "__main__":
    import threading
    
    # Create Gradio app
    gradio_app = create_gradio_app()
    
    # Mount Gradio app
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    
    # Start application
    logger.info("Starting Coqui TTS Web Interface...")
    logger.info("Web UI: http://localhost:2201")
    logger.info("API: http://localhost:8080")
    logger.info("TTS Server: http://localhost:5002")
    
    # Run startup event
    asyncio.run(startup_event())
    
    # Start servers
    def start_api_server():
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8080,
            log_level="info"
        )
    
    def start_tts_server():
        uvicorn.run(
            "modules.tts_handler:create_tts_server",
            host="0.0.0.0",
            port=5002,
            log_level="info"
        )
    
    def start_web_server():
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=2201,
            log_level="info"
        )
    
    # Start servers in separate threads
    api_thread = threading.Thread(target=start_api_server)
    tts_thread = threading.Thread(target=start_tts_server)
    web_thread = threading.Thread(target=start_web_server)
    
    api_thread.start()
    tts_thread.start()
    web_thread.start()
    
    # Wait for all threads
    api_thread.join()
    tts_thread.join()
    web_thread.join()