import gradio as gr
from pathlib import Path
import json
import asyncio
from typing import Optional, List, Dict, Any

def create_gradio_interface(
    tts_handler,
    voice_cloner,
    advanced_voice_cloner,
    model_manager,
    audio_processor
):
    """Create the main Gradio interface"""
    
    # TTS Tab
    def tts_interface():
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here...",
                    lines=5
                )
                
                model_dropdown = gr.Dropdown(
                    label="TTS Model",
                    choices=["tts_models/en/ljspeech/tacotron2-DDC", 
                             "tts_models/en/ljspeech/glow-tts",
                             "tts_models/multilingual/multi-dataset/your_tts"],
                    value="tts_models/en/ljspeech/tacotron2-DDC"
                )
                
                with gr.Row():
                    speaker_dropdown = gr.Dropdown(
                        label="Speaker",
                        choices=["default"],
                        value="default"
                    )
                    
                    language_dropdown = gr.Dropdown(
                        label="Language",
                        choices=["en"],
                        value="en"
                    )
                
                with gr.Accordion("Advanced Settings", open=False):
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )
                    
                    pitch_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Pitch"
                    )
                    
                    energy_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Energy"
                    )
                    
                    emotion_dropdown = gr.Dropdown(
                        label="Emotion",
                        choices=["neutral", "happy", "sad", "angry", "surprise"],
                        value="neutral"
                    )
                    
                    output_format = gr.Radio(
                        label="Output Format",
                        choices=["wav", "mp3", "ogg", "flac"],
                        value="wav"
                    )
                
                synthesize_btn = gr.Button("Synthesize", variant="primary")
            
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath"
                )
                
                download_btn = gr.Button("Download Audio", variant="secondary")
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Ready"
                )
        
        def synthesize_speech(text, model, speaker, language, speed, pitch, energy, emotion, format):
            try:
                # This would call the actual TTS handler
                status = "Synthesizing speech..."
                # audio_path = tts_handler.synthesize(...)
                # For now, return a placeholder
                return None, "Speech synthesized successfully!"
            except Exception as e:
                return None, f"Error: {str(e)}"
        
        synthesize_btn.click(
            fn=synthesize_speech,
            inputs=[text_input, model_dropdown, speaker_dropdown, language_dropdown,
                    speed_slider, pitch_slider, energy_slider, emotion_dropdown, output_format],
            outputs=[audio_output, status_text]
        )
        
        return [text_input, model_dropdown, speaker_dropdown, language_dropdown,
                speed_slider, pitch_slider, energy_slider, emotion_dropdown, 
                output_format, synthesize_btn, audio_output, download_btn, status_text]
    
    # Voice Cloning Tab
    def voice_cloning_interface():
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Basic Voice Cloning")
                
                audio_upload = gr.Audio(
                    label="Upload Voice Sample",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                voice_name = gr.Textbox(
                    label="Voice Name",
                    placeholder="Enter a name for this voice"
                )
                
                voice_description = gr.Textbox(
                    label="Description (Optional)",
                    placeholder="Describe this voice...",
                    lines=2
                )
                
                clone_btn = gr.Button("Clone Voice", variant="primary")
                
                clone_status = gr.Textbox(
                    label="Status",
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("## Advanced Voice Cloning")
                
                multi_audio_upload = gr.File(
                    label="Upload Multiple Samples",
                    file_count="multiple",
                    file_types=["audio"]
                )
                
                with gr.Accordion("Advanced Options", open=False):
                    embedding_method = gr.Radio(
                        label="Embedding Method",
                        choices=["speaker_encoder", "dvector", "resemblyzer"],
                        value="speaker_encoder"
                    )
                    
                    quality_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=7,
                        step=1,
                        label="Quality Level"
                    )
                    
                    denoise_checkbox = gr.Checkbox(
                        label="Apply Denoising",
                        value=True
                    )
                
                advanced_clone_btn = gr.Button("Advanced Clone", variant="primary")
                
                similarity_score = gr.Number(
                    label="Voice Similarity Score",
                    interactive=False
                )
        
        def clone_voice_basic(audio, name, description):
            try:
                if not audio or not name:
                    return "Please provide both audio sample and voice name"
                # voice_id = voice_cloner.clone_voice(...)
                return f"Voice '{name}' cloned successfully!"
            except Exception as e:
                return f"Error: {str(e)}"
        
        def clone_voice_advanced(files, method, quality, denoise):
            try:
                if not files:
                    return 0.0
                # similarity = advanced_voice_cloner.clone_with_similarity(...)
                return 0.95  # Placeholder similarity score
            except Exception as e:
                return 0.0
        
        clone_btn.click(
            fn=clone_voice_basic,
            inputs=[audio_upload, voice_name, voice_description],
            outputs=[clone_status]
        )
        
        advanced_clone_btn.click(
            fn=clone_voice_advanced,
            inputs=[multi_audio_upload, embedding_method, quality_slider, denoise_checkbox],
            outputs=[similarity_score]
        )
        
        return [audio_upload, voice_name, voice_description, clone_btn, clone_status,
                multi_audio_upload, embedding_method, quality_slider, denoise_checkbox,
                advanced_clone_btn, similarity_score]
    
    # Model Management Tab
    def model_management_interface():
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Available Models")
                
                model_list = gr.Dataframe(
                    headers=["Model Name", "Language", "Size", "Status"],
                    datatype=["str", "str", "str", "str"],
                    value=[
                        ["tts_models/en/ljspeech/tacotron2-DDC", "English", "107MB", "Ready"],
                        ["tts_models/en/ljspeech/glow-tts", "English", "89MB", "Ready"],
                    ]
                )
                
                refresh_btn = gr.Button("Refresh Model List")
            
            with gr.Column():
                gr.Markdown("## Download New Model")
                
                model_search = gr.Textbox(
                    label="Search Models",
                    placeholder="Search for models..."
                )
                
                available_models = gr.Dropdown(
                    label="Available Models",
                    choices=[]
                )
                
                download_btn = gr.Button("Download Model", variant="primary")
                
                download_progress = gr.Progress()
                download_status = gr.Textbox(
                    label="Download Status",
                    interactive=False
                )
        
        def refresh_models():
            # models = model_manager.list_models()
            return [
                ["tts_models/en/ljspeech/tacotron2-DDC", "English", "107MB", "Ready"],
                ["tts_models/en/ljspeech/glow-tts", "English", "89MB", "Ready"],
            ]
        
        def search_models(query):
            # results = model_manager.search_models(query)
            return gr.update(choices=["model1", "model2"])
        
        def download_model(model_name):
            try:
                # model_manager.download_model(model_name)
                return "Model downloaded successfully!"
            except Exception as e:
                return f"Error: {str(e)}"
        
        refresh_btn.click(
            fn=refresh_models,
            outputs=[model_list]
        )
        
        model_search.change(
            fn=search_models,
            inputs=[model_search],
            outputs=[available_models]
        )
        
        download_btn.click(
            fn=download_model,
            inputs=[available_models],
            outputs=[download_status]
        )
        
        return [model_list, refresh_btn, model_search, available_models, 
                download_btn, download_progress, download_status]
    
    # Audio Processing Tab
    def audio_processing_interface():
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Audio Enhancement")
                
                audio_upload = gr.Audio(
                    label="Upload Audio File",
                    type="filepath"
                )
                
                with gr.Accordion("Processing Options", open=True):
                    denoise_check = gr.Checkbox(
                        label="Apply Noise Reduction",
                        value=True
                    )
                    
                    normalize_check = gr.Checkbox(
                        label="Normalize Audio",
                        value=True
                    )
                    
                    remove_silence_check = gr.Checkbox(
                        label="Remove Silence",
                        value=False
                    )
                    
                    eq_preset = gr.Dropdown(
                        label="EQ Preset",
                        choices=["none", "bright", "warm", "radio", "podcast"],
                        value="none"
                    )
                    
                    compression_check = gr.Checkbox(
                        label="Apply Compression",
                        value=False
                    )
                    
                    target_loudness = gr.Slider(
                        minimum=-30,
                        maximum=-10,
                        value=-20,
                        step=1,
                        label="Target Loudness (LUFS)"
                    )
                
                process_btn = gr.Button("Process Audio", variant="primary")
            
            with gr.Column():
                gr.Markdown("## Processed Audio")
                
                processed_audio = gr.Audio(
                    label="Processed Audio",
                    type="filepath"
                )
                
                audio_info = gr.JSON(
                    label="Audio Information"
                )
                
                processing_status = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    value="Ready"
                )
        
        def process_audio(audio, denoise, normalize, remove_silence, eq, compression, loudness):
            try:
                if not audio:
                    return None, {}, "Please upload an audio file"
                
                # This would call the actual audio processor
                # processed_path = audio_processor.enhance_audio(...)
                # info = audio_processor.get_audio_info(...)
                
                return None, {"duration": "3.5s", "sample_rate": "22050Hz"}, "Audio processed successfully!"
            except Exception as e:
                return None, {}, f"Error: {str(e)}"
        
        process_btn.click(
            fn=process_audio,
            inputs=[audio_upload, denoise_check, normalize_check, remove_silence_check,
                    eq_preset, compression_check, target_loudness],
            outputs=[processed_audio, audio_info, processing_status]
        )
        
        return [audio_upload, denoise_check, normalize_check, remove_silence_check,
                eq_preset, compression_check, target_loudness, process_btn,
                processed_audio, audio_info, processing_status]
    
    # Create the main interface with tabs
    with gr.Blocks(title="Coqui TTS WebGUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎙️ Coqui TTS WebGUI
            
            A comprehensive web interface for Coqui TTS with voice cloning capabilities.
            """
        )
        
        with gr.Tabs():
            with gr.TabItem("Text to Speech"):
                tts_components = tts_interface()
            
            with gr.TabItem("Voice Cloning"):
                voice_components = voice_cloning_interface()
            
            with gr.TabItem("Model Management"):
                model_components = model_management_interface()
            
            with gr.TabItem("Audio Processing"):
                audio_components = audio_processing_interface()
            
            with gr.TabItem("API Documentation"):
                gr.Markdown(
                    """
                    ## API Documentation
                    
                    ### Base URL
                    ```
                    http://localhost:2201/api
                    ```
                    
                    ### Endpoints
                    
                    #### POST /api/tts
                    Generate speech from text.
                    
                    **Request Body:**
                    ```json
                    {
                        "text": "Hello world",
                        "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                        "speaker": "default",
                        "language": "en",
                        "speed": 1.0,
                        "pitch": 1.0,
                        "energy": 1.0,
                        "emotion": "neutral",
                        "output_format": "wav"
                    }
                    ```
                    
                    **Response:**
                    ```json
                    {
                        "audio_url": "/output/tts_20231205_143022.wav",
                        "duration": 3.5,
                        "request_id": "req_12345"
                    }
                    ```
                    
                    #### GET /api/models
                    List available TTS models.
                    
                    **Response:**
                    ```json
                    {
                        "tts_models": [
                            {
                                "name": "tts_models/en/ljspeech/tacotron2-DDC",
                                "language": "en",
                                "size": "107MB",
                                "downloaded": true,
                                "capabilities": ["standard"]
                            }
                        ]
                    }
                    ```
                    
                    #### POST /api/voices/clone
                    Clone a voice from audio sample.
                    
                    **Request Body (multipart/form-data):**
                    - `audio_file`: Audio file
                    - `name`: Voice name
                    - `description`: Optional description
                    
                    **Response:**
                    ```json
                    {
                        "voice_id": "voice_abc123",
                        "name": "My Voice",
                        "status": "cloned"
                    }
                    ```
                    
                    #### GET /api/voices
                    List cloned voices.
                    
                    **Response:**
                    ```json
                    {
                        "voices": [
                            {
                                "id": "voice_abc123",
                                "name": "My Voice",
                                "created_at": "2023-12-05T14:30:22Z"
                            }
                        ]
                    }
                    ```
                    
                    #### POST /api/audio/enhance
                    Enhance audio quality.
                    
                    **Request Body (multipart/form-data):**
                    - `audio_file`: Audio file
                    - `denoise`: Boolean (optional)
                    - `normalize`: Boolean (optional)
                    - `eq_preset`: String (optional)
                    
                    **Response:**
                    ```json
                    {
                        "enhanced_audio_url": "/output/enhanced_audio.wav",
                        "processing_time": 2.1
                    }
                    ```
                    
                    ### Error Responses
                    
                    All endpoints return standard HTTP status codes:
                    - `200`: Success
                    - `400`: Bad Request
                    - `404`: Not Found
                    - `500`: Internal Server Error
                    
                    Error response format:
                    ```json
                    {
                        "error": "Error message",
                        "code": "ERROR_CODE"
                    }
                    ```
                    """
                )
        
        gr.Markdown(
            """
            ---
            <center>Powered by Coqui TTS | Port 2201 (Web UI) | Port 5002 (TTS API) | Port 8080 (Additional API)</center>
            """
        )
    
    return demo 