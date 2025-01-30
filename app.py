import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
from medical_rag import MedicalRAGPipeline
import logging
from typing import Optional, Dict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioHandler:
    def __init__(self):
        """Initialize AudioHandler with error handling"""
        self.recognizer = sr.Recognizer()
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Initialized temporary directory at {self.temp_dir}")

    def process_audio_data(self, audio_bytes: bytes) -> Optional[str]:
        """Process audio data from audio-recorder-streamlit"""
        if not audio_bytes:
            return None

        try:
            # Save audio bytes to temporary file
            temp_file = os.path.join(self.temp_dir, "temp_audio.wav")
            with open(temp_file, 'wb') as f:
                f.write(audio_bytes)

            # Transcribe the audio
            with sr.AudioFile(temp_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                logger.info("Successfully transcribed audio")
                return text

        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def text_to_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech and return bytes"""
        if not text:
            return None

        try:
            # Create temporary file for audio
            temp_file = os.path.join(self.temp_dir, "temp_speech.mp3")
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file)

            # Read file as bytes
            with open(temp_file, 'rb') as f:
                audio_bytes = f.read()

            return audio_bytes

        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            return None
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def cleanup(self):
        """Clean up temporary directory"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")

class MedicalAssistantApp:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'audio_handler' not in st.session_state:
            st.session_state.audio_handler = AudioHandler()
        
        if 'pipeline' not in st.session_state:
            try:
                st.session_state.pipeline = MedicalRAGPipeline(
                    gemini_api_key="AIzaSyAulFzIkm9yvMawZBV5-HFoCEEu2BRzn7A"
                )
                # Load data and create index
                documents = st.session_state.pipeline.load_diseases_data("diseases.json")
                st.session_state.intents_data = st.session_state.pipeline.load_intents_data("intents.json")
                st.session_state.pipeline.create_index(documents)
            except Exception as e:
                logger.error(f"Error initializing pipeline: {e}")
                st.error("Error initializing the medical assistant. Please try refreshing the page.")

    def run(self):
        st.title("Medical Assistant")
        
        # Create tabs
        text_tab, audio_tab = st.tabs(["Text Input", "Audio Input"])

        # Text Input Tab
        with text_tab:
            self.handle_text_input()

        # Audio Input Tab
        with audio_tab:
            self.handle_audio_input()

        # Add disclaimer
        st.markdown("---")
        st.markdown("""
        **Disclaimer**: This medical assistant is for informational purposes only. 
        Always consult healthcare professionals for medical advice, diagnosis, or treatment.
        """)

    def handle_text_input(self):
        """Handle text input interactions"""
        text_query = st.text_input("Enter your medical question:")
        if st.button("Get Answer", key="text_button"):
            if text_query:
                try:
                    response = st.session_state.pipeline.generate_response(
                        text_query, 
                        st.session_state.intents_data
                    )
                    st.write("Response:", response)
                    
                    audio_bytes = st.session_state.audio_handler.text_to_speech(response)
                    if audio_bytes:
                        st.audio(audio_bytes, format='audio/mp3')
                except Exception as e:
                    logger.error(f"Error processing text input: {e}")
                    st.error("Error generating response. Please try again.")

    def handle_audio_input(self):
        """Handle audio input using audio-recorder-streamlit"""
        st.write("Click the microphone to start recording")
        audio_bytes = audio_recorder()

        if audio_bytes:
            try:
                # Process the audio
                text = st.session_state.audio_handler.process_audio_data(audio_bytes)
                if text:
                    st.write("You said:", text)
                    
                    # Generate response
                    response = st.session_state.pipeline.generate_response(
                        text,
                        st.session_state.intents_data
                    )
                    st.write("Response:", response)
                    
                    # Convert response to speech
                    audio_response = st.session_state.audio_handler.text_to_speech(response)
                    if audio_response:
                        st.audio(audio_response, format='audio/mp3')
                else:
                    st.error("Could not understand the audio. Please try again.")
            except Exception as e:
                logger.error(f"Error processing audio input: {e}")
                st.error("Error processing audio. Please try again.")

if __name__ == "__main__":
    app = MedicalAssistantApp()
    app.run()
