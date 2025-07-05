import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
from pydub.silence import detect_nonsilent
import noisereduce as nr
import pyloudnorm as pyln

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handler for audio processing and format conversion"""
    
    def __init__(self):
        self.sample_rate = 22050  # Default sample rate for TTS
        self.meter = pyln.Meter(self.sample_rate)  # For loudness normalization
        
    def enhance_audio(
        self,
        audio_path: str,
        denoise: bool = True,
        normalize: bool = True,
        remove_silence: bool = False,
        eq_preset: Optional[str] = None,
        compression: bool = False,
        target_loudness: float = -20.0  # LUFS
    ) -> str:
        """Enhance audio with multiple processing options"""
        try:
            logger.info(f"Enhancing audio: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Denoise
            if denoise:
                audio = self._denoise_audio(audio, sr)
            
            # Remove silence
            if remove_silence:
                audio = self._remove_silence(audio, sr)
            
            # Apply EQ
            if eq_preset:
                audio = self._apply_eq(audio, sr, eq_preset)
            
            # Apply compression
            if compression:
                audio = self._apply_compression(audio, sr)
            
            # Normalize
            if normalize:
                audio = self._normalize_loudness(audio, sr, target_loudness)
            
            # Save enhanced audio
            output_path = Path(audio_path).with_stem(f"{Path(audio_path).stem}_enhanced")
            sf.write(str(output_path), audio, sr)
            
            logger.info(f"Audio enhanced and saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error enhancing audio: {str(e)}")
            return audio_path
    
    def _denoise_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply noise reduction"""
        try:
            # Perform noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=True,
                prop_decrease=1.0
            )
            return reduced_noise
        except Exception as e:
            logger.warning(f"Denoising failed: {str(e)}")
            return audio
    
    def _remove_silence(
        self, 
        audio: np.ndarray, 
        sr: int,
        min_silence_len: int = 500,  # milliseconds
        silence_thresh: int = -40  # dB
    ) -> np.ndarray:
        """Remove silence from audio"""
        try:
            # Convert to AudioSegment for easier manipulation
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=sr,
                sample_width=audio.dtype.itemsize,
                channels=1
            )
            
            # Detect non-silent chunks
            nonsilent_chunks = detect_nonsilent(
                audio_segment,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh
            )
            
            if not nonsilent_chunks:
                return audio
            
            # Concatenate non-silent chunks
            processed = AudioSegment.empty()
            for start, end in nonsilent_chunks:
                processed += audio_segment[start:end]
            
            # Convert back to numpy
            return np.array(processed.get_array_of_samples(), dtype=np.float32) / 32768.0
            
        except Exception as e:
            logger.warning(f"Silence removal failed: {str(e)}")
            return audio
    
    def _apply_eq(self, audio: np.ndarray, sr: int, preset: str) -> np.ndarray:
        """Apply EQ preset"""
        try:
            # EQ presets for voice
            presets = {
                "bright": {
                    "frequencies": [200, 1000, 4000, 8000],
                    "gains": [-2, 0, 3, 2]
                },
                "warm": {
                    "frequencies": [100, 300, 1000, 5000],
                    "gains": [2, 3, 0, -2]
                },
                "radio": {
                    "frequencies": [100, 300, 2000, 8000],
                    "gains": [-5, 2, 3, -3]
                },
                "podcast": {
                    "frequencies": [80, 200, 2000, 10000],
                    "gains": [-3, 2, 1, 0]
                }
            }
            
            if preset not in presets:
                return audio
            
            eq_settings = presets[preset]
            
            # Apply parametric EQ
            processed = audio.copy()
            for freq, gain in zip(eq_settings["frequencies"], eq_settings["gains"]):
                processed = self._apply_band_eq(processed, sr, freq, gain)
            
            return processed
            
        except Exception as e:
            logger.warning(f"EQ application failed: {str(e)}")
            return audio
    
    def _apply_band_eq(
        self, 
        audio: np.ndarray, 
        sr: int, 
        center_freq: float, 
        gain_db: float,
        q: float = 1.0
    ) -> np.ndarray:
        """Apply parametric EQ to a specific frequency band"""
        try:
            # Convert gain from dB to linear
            gain_linear = 10 ** (gain_db / 20)
            
            # Design peaking EQ filter
            w0 = 2 * np.pi * center_freq / sr
            alpha = np.sin(w0) / (2 * q)
            
            # Peaking EQ coefficients
            b0 = 1 + alpha * gain_linear
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * gain_linear
            a0 = 1 + alpha / gain_linear
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / gain_linear
            
            # Normalize coefficients
            b = [b0/a0, b1/a0, b2/a0]
            a = [1, a1/a0, a2/a0]
            
            # Apply filter
            return scipy.signal.lfilter(b, a, audio)
            
        except Exception as e:
            logger.warning(f"Band EQ failed: {str(e)}")
            return audio
    
    def _apply_compression(
        self, 
        audio: np.ndarray, 
        sr: int,
        threshold: float = -20.0,  # dB
        ratio: float = 4.0,
        attack: float = 5.0,  # ms
        release: float = 50.0  # ms
    ) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Convert to AudioSegment for compression
            audio_segment = AudioSegment(
                audio.tobytes(),
                frame_rate=sr,
                sample_width=audio.dtype.itemsize,
                channels=1
            )
            
            # Apply compression
            compressed = compress_dynamic_range(
                audio_segment,
                threshold=threshold,
                ratio=ratio,
                attack=attack,
                release=release
            )
            
            # Convert back to numpy
            return np.array(compressed.get_array_of_samples(), dtype=np.float32) / 32768.0
            
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
            return audio
    
    def _normalize_loudness(
        self, 
        audio: np.ndarray, 
        sr: int, 
        target_loudness: float = -20.0
    ) -> np.ndarray:
        """Normalize audio to target loudness (LUFS)"""
        try:
            # Measure loudness
            loudness = self.meter.integrated_loudness(audio)
            
            # Calculate normalization gain
            gain = target_loudness - loudness
            
            # Apply gain
            normalized = audio * (10 ** (gain / 20))
            
            # Prevent clipping
            if np.max(np.abs(normalized)) > 0.95:
                normalized = normalized / np.max(np.abs(normalized)) * 0.95
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Loudness normalization failed: {str(e)}")
            return audio
    
    def get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            return len(audio) / sr
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return 0.0
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get detailed audio information"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            
            info = {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "channels": 1,  # librosa loads as mono by default
                "format": Path(audio_path).suffix.lower(),
                "file_size": Path(audio_path).stat().st_size,
                "bit_depth": 32,  # librosa loads as float32
                "loudness": self.meter.integrated_loudness(audio),
                "peak_level": 20 * np.log10(np.max(np.abs(audio))),
                "rms_level": 20 * np.log10(np.sqrt(np.mean(audio**2))),
                "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
                "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(audio))
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return {}
    
    def convert_format(
        self,
        audio_path: str,
        output_format: str,
        sample_rate: Optional[int] = None,
        bitrate: Optional[str] = None
    ) -> str:
        """Convert audio to different format"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=sample_rate)
            
            # Prepare output path
            output_path = Path(audio_path).with_suffix(f".{output_format.lower()}")
            
            # Convert using soundfile for lossless formats
            if output_format.lower() in ['wav', 'flac', 'ogg']:
                sf.write(str(output_path), audio, sr)
            else:
                # Use pydub for lossy formats
                audio_segment = AudioSegment(
                    audio.tobytes(),
                    frame_rate=sr,
                    sample_width=audio.dtype.itemsize,
                    channels=1
                )
                
                export_params = {}
                if bitrate:
                    export_params['bitrate'] = bitrate
                
                audio_segment.export(str(output_path), format=output_format, **export_params)
            
            logger.info(f"Audio converted to {output_format}: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error converting audio format: {str(e)}")
            return audio_path
    
    def split_audio(
        self,
        audio_path: str,
        segment_duration: float = 30.0  # seconds
    ) -> List[str]:
        """Split audio into segments"""
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            total_duration = len(audio) / sr
            
            segments = []
            start_time = 0
            segment_idx = 0
            
            while start_time < total_duration:
                end_time = min(start_time + segment_duration, total_duration)
                
                # Extract segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment = audio[start_sample:end_sample]
                
                # Save segment
                segment_path = Path(audio_path).with_stem(
                    f"{Path(audio_path).stem}_segment_{segment_idx:03d}"
                )
                sf.write(str(segment_path), segment, sr)
                segments.append(str(segment_path))
                
                start_time = end_time
                segment_idx += 1
            
            logger.info(f"Audio split into {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error splitting audio: {str(e)}")
            return [audio_path]
    
    def merge_audio(
        self,
        audio_paths: List[str],
        output_path: str,
        crossfade_duration: float = 0.1  # seconds
    ) -> str:
        """Merge multiple audio files"""
        try:
            if not audio_paths:
                raise ValueError("No audio files to merge")
            
            # Load first audio file
            merged_audio, sr = librosa.load(audio_paths[0], sr=None)
            
            # Merge remaining files
            for audio_path in audio_paths[1:]:
                audio, _ = librosa.load(audio_path, sr=sr)
                
                # Apply crossfade if specified
                if crossfade_duration > 0:
                    crossfade_samples = int(crossfade_duration * sr)
                    if len(merged_audio) > crossfade_samples and len(audio) > crossfade_samples:
                        # Fade out end of merged audio
                        fade_out = np.linspace(1, 0, crossfade_samples)
                        merged_audio[-crossfade_samples:] *= fade_out
                        
                        # Fade in start of new audio
                        fade_in = np.linspace(0, 1, crossfade_samples)
                        audio[:crossfade_samples] *= fade_in
                        
                        # Overlap and add
                        merged_audio[-crossfade_samples:] += audio[:crossfade_samples]
                        merged_audio = np.concatenate([merged_audio, audio[crossfade_samples:]])
                    else:
                        merged_audio = np.concatenate([merged_audio, audio])
                else:
                    merged_audio = np.concatenate([merged_audio, audio])
            
            # Save merged audio
            sf.write(output_path, merged_audio, sr)
            
            logger.info(f"Audio files merged: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error merging audio: {str(e)}")
            return ""
    
    async def apply_effects(
        self,
        audio_path: Path,
        effects: Dict[str, Any]
    ) -> Path:
        """Apply various audio effects"""
        try:
            enhanced_path = self.enhance_audio(
                str(audio_path),
                denoise=effects.get('denoise', False),
                normalize=effects.get('normalize', True),
                remove_silence=effects.get('remove_silence', False),
                eq_preset=effects.get('eq_preset'),
                compression=effects.get('compression', False),
                target_loudness=effects.get('target_loudness', -20.0)
            )
            
            return Path(enhanced_path)
            
        except Exception as e:
            logger.error(f"Error applying effects: {str(e)}")
            return audio_path