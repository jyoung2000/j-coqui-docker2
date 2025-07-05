import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torchaudio
from TTS.api import TTS
from TTS.tts.utils.synthesis import synthesis
from TTS.config import load_config
from TTS.tts.models import setup_model
import librosa
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class AdvancedVoiceCloner:
    def __init__(self):
        self.voice_samples_path = Path("/app/voice_samples")
        self.voice_samples_path.mkdir(exist_ok=True)
        self.voice_db_path = Path("/app/user_data/voices.json")
        self.voice_encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_voice_database()
        
    def _load_voice_database(self):
        """Load voice database"""
        if self.voice_db_path.exists():
            with open(self.voice_db_path, 'r') as f:
                self.voice_db = json.load(f)
        else:
            self.voice_db = {}
            
    def _save_voice_database(self):
        """Save voice database"""
        self.voice_db_path.parent.mkdir(exist_ok=True, parents=True)
        with open(self.voice_db_path, 'w') as f:
            json.dump(self.voice_db, f, indent=2)
    
    async def clone_voice(
        self,
        audio_path: str,
        name: str,
        description: Optional[str] = None,
        quality: int = 7,
        denoise: bool = True,
        enhance: bool = True,
        multiple_samples: Optional[List[str]] = None
    ) -> str:
        """Clone a voice with advanced options"""
        try:
            voice_id = str(uuid.uuid4())
            logger.info(f"Starting advanced voice cloning for '{name}' (ID: {voice_id})")
            
            # Process audio samples
            if multiple_samples:
                # Process multiple samples for better quality
                processed_samples = []
                for sample_path in multiple_samples:
                    processed = await self._process_audio_sample(
                        sample_path, 
                        denoise=denoise,
                        enhance=enhance
                    )
                    processed_samples.append(processed)
                
                # Combine samples
                combined_path = await self._combine_audio_samples(processed_samples)
                main_audio_path = combined_path
            else:
                # Process single sample
                main_audio_path = await self._process_audio_sample(
                    audio_path,
                    denoise=denoise,
                    enhance=enhance
                )
            
            # Extract voice characteristics
            voice_features = await self._extract_voice_features(main_audio_path, quality)
            
            # Save voice sample
            voice_filename = f"{voice_id}.wav"
            voice_path = self.voice_samples_path / voice_filename
            
            # Copy processed audio to voice samples
            import shutil
            shutil.copy(main_audio_path, voice_path)
            
            # Create voice profile
            voice_profile = {
                "id": voice_id,
                "name": name,
                "description": description or f"Voice cloned on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "file_path": str(voice_path),
                "created_at": datetime.now().isoformat(),
                "quality": quality,
                "features": voice_features,
                "sample_rate": voice_features.get("sample_rate", 22050),
                "duration": voice_features.get("duration", 0),
                "settings": {
                    "denoise": denoise,
                    "enhance": enhance,
                    "quality_level": quality
                }
            }
            
            # Save to database
            self.voice_db[voice_id] = voice_profile
            self._save_voice_database()
            
            logger.info(f"Voice '{name}' cloned successfully with ID: {voice_id}")
            return voice_id
            
        except Exception as e:
            logger.error(f"Advanced voice cloning error: {str(e)}")
            raise
    
    async def _process_audio_sample(
        self, 
        audio_path: str, 
        denoise: bool = True,
        enhance: bool = True
    ) -> str:
        """Process audio sample for voice cloning"""
        try:
            # Load audio
            waveform, sample_rate = librosa.load(audio_path, sr=22050)
            
            # Denoise if requested
            if denoise:
                waveform = self._denoise_audio(waveform, sample_rate)
            
            # Enhance if requested
            if enhance:
                waveform = self._enhance_audio(waveform, sample_rate)
            
            # Normalize
            waveform = librosa.util.normalize(waveform)
            
            # Save processed audio
            processed_path = Path(f"/app/temp/processed_{os.path.basename(audio_path)}")
            processed_path.parent.mkdir(exist_ok=True, parents=True)
            sf.write(processed_path, waveform, sample_rate)
            
            return str(processed_path)
            
        except Exception as e:
            logger.error(f"Error processing audio sample: {str(e)}")
            raise
    
    def _denoise_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply denoising to audio"""
        try:
            # Simple spectral gating for noise reduction
            # Get noise profile from first 0.5 seconds
            noise_sample = waveform[:int(0.5 * sample_rate)]
            
            # Compute spectrum
            D = librosa.stft(waveform)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Estimate noise spectrum
            noise_D = librosa.stft(noise_sample)
            noise_magnitude = np.abs(noise_D)
            noise_profile = np.median(noise_magnitude, axis=1, keepdims=True)
            
            # Spectral subtraction
            cleaned_magnitude = magnitude - noise_profile
            cleaned_magnitude = np.maximum(cleaned_magnitude, 0)
            
            # Reconstruct
            cleaned_D = cleaned_magnitude * np.exp(1j * phase)
            cleaned_waveform = librosa.istft(cleaned_D)
            
            return cleaned_waveform
            
        except Exception as e:
            logger.warning(f"Denoising failed: {str(e)}")
            return waveform
    
    def _enhance_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance audio quality"""
        try:
            # Apply gentle compression
            threshold = 0.7
            ratio = 3.0
            waveform = np.where(
                np.abs(waveform) > threshold,
                np.sign(waveform) * (threshold + (np.abs(waveform) - threshold) / ratio),
                waveform
            )
            
            # Apply EQ boost in speech frequencies
            # Boost 1-4 kHz range slightly
            sos = librosa.effects.preemphasis(waveform)
            
            return sos
            
        except Exception as e:
            logger.warning(f"Enhancement failed: {str(e)}")
            return waveform
    
    async def _combine_audio_samples(self, sample_paths: List[str]) -> str:
        """Combine multiple audio samples"""
        try:
            combined_waveform = []
            target_sr = 22050
            
            for path in sample_paths:
                waveform, sr = librosa.load(path, sr=target_sr)
                combined_waveform.append(waveform)
                
                # Add small silence between samples
                silence = np.zeros(int(0.5 * target_sr))
                combined_waveform.append(silence)
            
            # Concatenate all
            combined = np.concatenate(combined_waveform)
            
            # Save combined audio
            combined_path = Path(f"/app/temp/combined_{uuid.uuid4().hex}.wav")
            combined_path.parent.mkdir(exist_ok=True, parents=True)
            sf.write(combined_path, combined, target_sr)
            
            return str(combined_path)
            
        except Exception as e:
            logger.error(f"Error combining samples: {str(e)}")
            raise
    
    async def _extract_voice_features(self, audio_path: str, quality: int) -> Dict[str, Any]:
        """Extract detailed voice features"""
        try:
            # Load audio
            waveform, sample_rate = librosa.load(audio_path, sr=None)
            
            features = {
                "sample_rate": sample_rate,
                "duration": len(waveform) / sample_rate
            }
            
            # Extract pitch characteristics
            pitches, magnitudes = librosa.piptrack(y=waveform, sr=sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features["pitch_mean"] = np.mean(pitch_values)
                features["pitch_std"] = np.std(pitch_values)
                features["pitch_min"] = np.min(pitch_values)
                features["pitch_max"] = np.max(pitch_values)
            
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
            features["spectral_centroid_mean"] = np.mean(spectral_centroids)
            features["spectral_centroid_std"] = np.std(spectral_centroids)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            features["mfcc_std"] = np.std(mfccs, axis=1).tolist()
            
            # Extract voice embedding if encoder is available
            embedding = await self._get_voice_embedding(audio_path)
            if embedding is not None:
                features["voice_embedding"] = embedding.tolist()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {str(e)}")
            return {"sample_rate": 22050, "duration": 0}
    
    async def _get_voice_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Get voice embedding using Resemblyzer"""
        try:
            if self.voice_encoder is None:
                self.voice_encoder = VoiceEncoder()
            
            # Load and preprocess audio
            wav = preprocess_wav(audio_path)
            
            # Get embedding
            embedding = self.voice_encoder.embed_utterance(wav)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Voice embedding extraction failed: {str(e)}")
            return None
    
    def get_voice_similarity(self, voice_id1: str, voice_id2: str) -> float:
        """Calculate similarity between two voices"""
        try:
            voice1 = self.voice_db.get(voice_id1)
            voice2 = self.voice_db.get(voice_id2)
            
            if not voice1 or not voice2:
                return 0.0
            
            # Get embeddings
            embedding1 = voice1.get("features", {}).get("voice_embedding")
            embedding2 = voice2.get("features", {}).get("voice_embedding")
            
            if not embedding1 or not embedding2:
                return 0.0
            
            # Calculate cosine similarity
            embedding1 = np.array(embedding1)
            embedding2 = np.array(embedding2)
            
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating voice similarity: {str(e)}")
            return 0.0
    
    def get_voice_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get voice information"""
        return self.voice_db.get(voice_id)
    
    def list_voices(self) -> List[Dict[str, Any]]:
        """List all cloned voices"""
        voices = []
        for voice_id, voice_data in self.voice_db.items():
            voice_info = voice_data.copy()
            voice_info["id"] = voice_id
            # Don't include large embedding data in list
            if "features" in voice_info and "voice_embedding" in voice_info["features"]:
                voice_info["features"] = voice_info["features"].copy()
                del voice_info["features"]["voice_embedding"]
            voices.append(voice_info)
        
        # Sort by creation date (newest first)
        voices.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return voices
    
    def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice"""
        try:
            if voice_id in self.voice_db:
                voice_data = self.voice_db[voice_id]
                
                # Delete audio file
                file_path = Path(voice_data.get("file_path", ""))
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from database
                del self.voice_db[voice_id]
                self._save_voice_database()
                
                logger.info(f"Voice {voice_id} deleted successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting voice: {str(e)}")
            return False