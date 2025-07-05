import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator
import torch
import torchaudio
from TTS.api import TTS
import numpy as np
import soundfile as sf
from datetime import datetime

logger = logging.getLogger(__name__)

class TTSHandler:
    """Handler for TTS synthesis operations"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TTS Handler initialized with device: {self.device}")
        
    def load_model(self, model_name: str) -> TTS:
        """Load a TTS model"""
        if model_name not in self.models:
            logger.info(f"Loading model: {model_name}")
            try:
                self.models[model_name] = TTS(model_name).to(self.device)
                logger.info(f"Model {model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise
        return self.models[model_name]
    
    async def synthesize(
        self,
        text: str,
        model_name: Optional[str] = None,
        speaker: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        language: Optional[str] = "en",
        speed: float = 1.0,
        pitch: float = 1.0,
        energy: float = 1.0,
        emotion: Optional[str] = None,
        output_format: str = "wav",
        stream: bool = False
    ) -> Path:
        """Synthesize speech from text with advanced options"""
        try:
            # Default model if not specified
            if not model_name:
                model_name = "tts_models/en/ljspeech/tacotron2-DDC"
            
            # Load model
            tts = self.load_model(model_name)
            
            # Prepare output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"tts_{timestamp}.{output_format}"
            output_path = Path("/app/output") / output_filename
            
            # Check if model supports voice cloning
            if speaker_wav and self._supports_voice_cloning(model_name):
                # Voice cloning synthesis
                logger.info(f"Synthesizing with voice cloning: {text[:50]}...")
                
                # Handle multi-lingual models
                if "multilingual" in model_name or "your_tts" in model_name or "xtts" in model_name:
                    tts.tts_to_file(
                        text=text,
                        speaker_wav=speaker_wav,
                        language=language,
                        file_path=str(output_path)
                    )
                else:
                    # Single language voice cloning
                    tts.tts_to_file(
                        text=text,
                        speaker_wav=speaker_wav,
                        file_path=str(output_path)
                    )
            
            # Multi-speaker synthesis
            elif speaker and self._is_multi_speaker(model_name):
                logger.info(f"Synthesizing with speaker {speaker}: {text[:50]}...")
                tts.tts_to_file(
                    text=text,
                    speaker=speaker,
                    language=language if self._is_multilingual(model_name) else None,
                    file_path=str(output_path)
                )
            
            # Standard synthesis
            else:
                logger.info(f"Standard synthesis: {text[:50]}...")
                tts.tts_to_file(
                    text=text,
                    file_path=str(output_path)
                )
            
            # Apply post-processing if needed
            if speed != 1.0 or pitch != 1.0:
                output_path = await self._apply_effects(
                    output_path, 
                    speed=speed, 
                    pitch=pitch
                )
            
            # Convert format if needed
            if output_format != "wav":
                output_path = await self._convert_format(output_path, output_format)
            
            logger.info(f"Synthesis completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Synthesis error: {str(e)}")
            raise
    
    async def stream_synthesis(
        self,
        text: str,
        model_name: Optional[str] = None,
        speaker: Optional[str] = None,
        language: Optional[str] = "en",
        chunk_size: int = 1024
    ) -> AsyncGenerator[bytes, None]:
        """Stream TTS synthesis for real-time applications"""
        try:
            if not model_name:
                model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            
            # Check if model supports streaming
            if not self._supports_streaming(model_name):
                # Fallback to regular synthesis
                audio_path = await self.synthesize(
                    text=text,
                    model_name=model_name,
                    speaker=speaker,
                    language=language
                )
                
                # Stream the file
                with open(audio_path, 'rb') as f:
                    while chunk := f.read(chunk_size):
                        yield chunk
                return
            
            # Load model for streaming
            tts = self.load_model(model_name)
            
            # Get speaker embeddings if needed
            if self._is_multi_speaker(model_name) and speaker:
                gpt_cond_latent, speaker_embedding = self._get_speaker_embeddings(
                    tts, speaker
                )
            else:
                gpt_cond_latent, speaker_embedding = None, None
            
            # Stream synthesis
            logger.info(f"Starting streaming synthesis: {text[:50]}...")
            
            # For XTTS streaming
            if "xtts" in model_name:
                chunks = tts.tts_model.inference_stream(
                    text,
                    language,
                    gpt_cond_latent,
                    speaker_embedding,
                    stream_chunk_size=20
                )
                
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        logger.info("First chunk generated")
                    
                    # Convert tensor to bytes
                    audio_bytes = (chunk.cpu().numpy() * 32767).astype(np.int16).tobytes()
                    yield audio_bytes
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            raise
    
    def _supports_voice_cloning(self, model_name: str) -> bool:
        """Check if model supports voice cloning"""
        voice_cloning_models = [
            "your_tts", "xtts", "bark", "tortoise"
        ]
        return any(vc_model in model_name for vc_model in voice_cloning_models)
    
    def _supports_streaming(self, model_name: str) -> bool:
        """Check if model supports streaming"""
        streaming_models = ["xtts"]
        return any(s_model in model_name for s_model in streaming_models)
    
    def _is_multi_speaker(self, model_name: str) -> bool:
        """Check if model is multi-speaker"""
        # Most models with these keywords support multiple speakers
        multi_speaker_keywords = [
            "vctk", "libritts", "multi", "your_tts", "xtts", "bark"
        ]
        return any(keyword in model_name.lower() for keyword in multi_speaker_keywords)
    
    def _is_multilingual(self, model_name: str) -> bool:
        """Check if model is multilingual"""
        return "multilingual" in model_name or "multi-dataset" in model_name
    
    def _get_speaker_embeddings(self, tts: TTS, speaker: str):
        """Get speaker embeddings for streaming"""
        # This is a simplified version - actual implementation depends on model
        try:
            if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_model'):
                # For XTTS models
                speaker_wav_path = f"/app/voice_samples/{speaker}.wav"
                if os.path.exists(speaker_wav_path):
                    return tts.synthesizer.tts_model.get_conditioning_latents(
                        audio_path=[speaker_wav_path]
                    )
        except Exception as e:
            logger.error(f"Error getting speaker embeddings: {str(e)}")
        
        return None, None
    
    async def _apply_effects(
        self, 
        audio_path: Path, 
        speed: float = 1.0, 
        pitch: float = 1.0
    ) -> Path:
        """Apply speed and pitch effects to audio"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Apply speed change
            if speed != 1.0:
                waveform = torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    int(sample_rate * speed)
                )
                sample_rate = int(sample_rate * speed)
            
            # Apply pitch shift (simplified)
            if pitch != 1.0:
                n_steps = int(12 * np.log2(pitch))
                if n_steps != 0:
                    # This is a simplified pitch shift - in reality you'd use a more sophisticated method
                    waveform = torchaudio.functional.pitch_shift(
                        waveform, sample_rate, n_steps
                    )
            
            # Save modified audio
            effects_path = audio_path.with_stem(f"{audio_path.stem}_effects")
            torchaudio.save(str(effects_path), waveform, sample_rate)
            
            return effects_path
            
        except Exception as e:
            logger.error(f"Error applying effects: {str(e)}")
            return audio_path
    
    async def _convert_format(self, audio_path: Path, target_format: str) -> Path:
        """Convert audio to target format"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert format
            output_path = audio_path.with_suffix(f".{target_format}")
            torchaudio.save(str(output_path), waveform, sample_rate)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting format: {str(e)}")
            return audio_path
    
    def get_available_speakers(self, model_name: str) -> list:
        """Get available speakers for a model"""
        try:
            if model_name in self.models:
                tts = self.models[model_name]
                if hasattr(tts, 'speakers'):
                    return tts.speakers
                elif hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_speakers'):
                    return tts.synthesizer.tts_speakers
            return []
        except Exception as e:
            logger.error(f"Error getting speakers: {str(e)}")
            return []
    
    def get_available_languages(self, model_name: str) -> list:
        """Get available languages for a model"""
        try:
            if model_name in self.models:
                tts = self.models[model_name]
                if hasattr(tts, 'languages'):
                    return tts.languages
                elif hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_languages'):
                    return tts.synthesizer.tts_languages
            return ["en"]
        except Exception as e:
            logger.error(f"Error getting languages: {str(e)}")
            return ["en"]