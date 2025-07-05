import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import requests
from TTS.api import TTS
from TTS.utils.manage import ModelManager as CoquiModelManager
import torch

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.models_path = Path("/app/models")
        self.models_path.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.model_info_cache = {}
        self.coqui_manager = CoquiModelManager()
        
    def list_all_models(self) -> List[Dict[str, Any]]:
        """List all available TTS models with detailed information"""
        try:
            # Get all available models from Coqui
            all_models = TTS.list_models()
            
            # Organize models by type
            organized_models = {
                "tts_models": [],
                "vocoder_models": [],
                "voice_conversion_models": []
            }
            
            for model in all_models:
                model_info = self._parse_model_name(model)
                
                # Add additional metadata
                model_data = {
                    "name": model,
                    "type": model_info["type"],
                    "language": model_info.get("language", "unknown"),
                    "dataset": model_info.get("dataset", "unknown"),
                    "model_name": model_info.get("model_name", "unknown"),
                    "downloaded": self._is_model_downloaded(model),
                    "size": self._get_model_size(model),
                    "description": self._get_model_description(model),
                    "capabilities": self._get_model_capabilities(model)
                }
                
                if model.startswith("tts_models"):
                    organized_models["tts_models"].append(model_data)
                elif model.startswith("vocoder_models"):
                    organized_models["vocoder_models"].append(model_data)
                elif model.startswith("voice_conversion"):
                    organized_models["voice_conversion_models"].append(model_data)
            
            return organized_models
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return {"tts_models": [], "vocoder_models": [], "voice_conversion_models": []}
    
    def _parse_model_name(self, model_name: str) -> Dict[str, str]:
        """Parse model name to extract components"""
        parts = model_name.split("/")
        
        result = {"full_name": model_name}
        
        if len(parts) >= 1:
            result["type"] = parts[0]
        if len(parts) >= 2:
            result["language"] = parts[1]
        if len(parts) >= 3:
            result["dataset"] = parts[2]
        if len(parts) >= 4:
            result["model_name"] = parts[3]
            
        return result
    
    def _is_model_downloaded(self, model_name: str) -> bool:
        """Check if model is already downloaded"""
        try:
            # Check local models directory
            model_path = self.models_path / model_name.replace("/", "_")
            if model_path.exists():
                return True
            
            # Check Coqui's default location
            home = Path.home()
            coqui_path = home / ".local" / "share" / "tts"
            model_folder = model_name.replace("/", "--")
            
            return (coqui_path / model_folder).exists()
            
        except Exception:
            return False
    
    def _get_model_size(self, model_name: str) -> str:
        """Get approximate model size"""
        # Model size estimates (you can expand this)
        size_map = {
            "tacotron2": "107MB",
            "glow-tts": "89MB",
            "vits": "145MB",
            "xtts": "1.8GB",
            "bark": "3.5GB",
            "tortoise": "4.2GB",
            "your_tts": "378MB",
            "hifigan": "45MB",
            "multiband-melgan": "28MB",
            "univnet": "58MB"
        }
        
        for key, size in size_map.items():
            if key in model_name.lower():
                return size
        
        return "Unknown"
    
    def _get_model_description(self, model_name: str) -> str:
        """Get model description"""
        descriptions = {
            "tacotron2": "Classic TTS model with good quality and speed",
            "glow-tts": "Flow-based model with fast inference",
            "vits": "End-to-end model with built-in vocoder",
            "xtts": "Advanced multilingual model with voice cloning",
            "bark": "Generative model for expressive speech",
            "tortoise": "High-quality model with voice cloning",
            "your_tts": "Multilingual model with zero-shot voice cloning",
            "hifigan": "High-quality neural vocoder",
            "multiband-melgan": "Fast and lightweight vocoder",
            "univnet": "Universal neural vocoder"
        }
        
        for key, desc in descriptions.items():
            if key in model_name.lower():
                return desc
        
        return "Text-to-speech model"
    
    def _get_model_capabilities(self, model_name: str) -> List[str]:
        """Get model capabilities"""
        capabilities = []
        
        # Check for voice cloning
        if any(vc in model_name for vc in ["xtts", "your_tts", "bark", "tortoise"]):
            capabilities.append("voice_cloning")
        
        # Check for multi-speaker
        if any(ms in model_name for ms in ["vctk", "libritts", "multi"]):
            capabilities.append("multi_speaker")
        
        # Check for multilingual
        if "multilingual" in model_name or "multi-dataset" in model_name:
            capabilities.append("multilingual")
        
        # Check for streaming
        if "xtts" in model_name:
            capabilities.append("streaming")
        
        # Check for emotion control
        if any(em in model_name for em in ["emotional", "expressive", "bark"]):
            capabilities.append("emotion_control")
        
        return capabilities
    
    async def download_model(self, model_name: str, model_type: str = "tts"):
        """Download a TTS model"""
        try:
            logger.info(f"Starting download of model: {model_name}")
            
            # Use TTS API to download
            if model_type == "tts":
                tts = TTS(model_name, progress_bar=True)
                logger.info(f"Model {model_name} downloaded successfully")
            
            # Save download info
            download_info = {
                "model_name": model_name,
                "download_date": datetime.now().isoformat(),
                "model_type": model_type
            }
            
            info_path = self.models_path / f"{model_name.replace('/', '_')}_info.json"
            with open(info_path, 'w') as f:
                json.dump(download_info, f)
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        try:
            # Check cache first
            if model_name in self.model_info_cache:
                return self.model_info_cache[model_name]
            
            info = {
                "name": model_name,
                "type": self._parse_model_name(model_name)["type"],
                "downloaded": self._is_model_downloaded(model_name),
                "size": self._get_model_size(model_name),
                "description": self._get_model_description(model_name),
                "capabilities": self._get_model_capabilities(model_name)
            }
            
            # Get additional info if model is loaded
            if model_name in self.loaded_models:
                model = self.loaded_models[model_name]
                if hasattr(model, 'speakers'):
                    info["speakers"] = model.speakers
                if hasattr(model, 'languages'):
                    info["languages"] = model.languages
            
            # Cache the info
            self.model_info_cache[model_name] = info
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"name": model_name, "error": str(e)}
    
    def load_default_models(self):
        """Load default models on startup"""
        default_models = [
            "tts_models/en/ljspeech/tacotron2-DDC",
            "tts_models/multilingual/multi-dataset/your_tts"
        ]
        
        for model in default_models:
            try:
                if self._is_model_downloaded(model):
                    logger.info(f"Loading default model: {model}")
                    self.loaded_models[model] = TTS(model)
            except Exception as e:
                logger.warning(f"Could not load default model {model}: {str(e)}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about models"""
        all_models = self.list_all_models()
        
        total_models = (
            len(all_models["tts_models"]) + 
            len(all_models["vocoder_models"]) + 
            len(all_models["voice_conversion_models"])
        )
        
        downloaded_models = sum(
            1 for model_list in all_models.values() 
            for model in model_list 
            if model["downloaded"]
        )
        
        # Get capabilities distribution
        capabilities_count = {}
        for model_list in all_models.values():
            for model in model_list:
                for capability in model["capabilities"]:
                    capabilities_count[capability] = capabilities_count.get(capability, 0) + 1
        
        return {
            "total_models": total_models,
            "downloaded_models": downloaded_models,
            "loaded_models": len(self.loaded_models),
            "tts_models": len(all_models["tts_models"]),
            "vocoder_models": len(all_models["vocoder_models"]),
            "voice_conversion_models": len(all_models["voice_conversion_models"]),
            "capabilities_distribution": capabilities_count
        }
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """Search for models matching query"""
        try:
            all_models = self.list_all_models()
            results = []
            
            query_lower = query.lower()
            
            for model_list in all_models.values():
                for model in model_list:
                    # Search in name, description, and capabilities
                    if (
                        query_lower in model["name"].lower() or
                        query_lower in model["description"].lower() or
                        any(query_lower in cap for cap in model["capabilities"])
                    ):
                        results.append(model)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching models: {str(e)}")
            return []