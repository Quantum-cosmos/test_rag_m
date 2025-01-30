import streamlit as st
from medical_rag import MedicalRAGPipeline
from audio_utils import AudioHandler
import json
import os
import logging
from typing import Optional, Dict
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalAssistantApp:
    def __init__(self):
        """Initialize the application with error handling"""
        self.initialize_session_state()
        self.setup_components()

    def initialize_session_state(self) -> None:
        """Initialize session state variables"""
        if 'audio_handler' not in st.session_state:
            st.session_state.audio_handler = AudioHandler()
        
        if 'pipeline' not in st.session_state:
            try:
                st.session_state.pipeline = MedicalRAGPipeline(
                    gemini_api_key=os.getenv('GEMINI_API_KEY')
                )
                # Load data and create index
                self.load_data()
            except Exception as e:
                logger.error(f"Error initializing pipeline: {e}")
                st.error("Error initializing the medical assistant. Please try refreshing the page.")

    def load_data(self) -> None:
        """Load necessary data with error handling"""
        try:
            documents = st.session_state.pipeline.load_diseases_data("diseases.json")
            st.session_state.intents_data = st.session_state.pipeline.load_intents_data("intents.json")
            st.session_state.pipeline.create_index(documents)
            logger.info("Successfully loaded all data")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error("Error loading medical data. Some features may be limited.")

    def setup_components(self) -> None:
        """Setup UI components"""
        st.title("Medical Assistant")
        st.write("Choose your preferred interaction method:")

    def handle_text_input(self) -> None:
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
                    
                    audio_file = st.session_state.audio_handler.text_to_speech(response)
                    if audio_file:
                        st.audio(audio_file)
                except Exception as e:
                    logger.error(f"Error processing text input: {e}")
                    st.error("Error generating response. Please try again.")

    def handle_mic_input(self) -> None:
        """Handle microphone input interactions"""
        if st.button("Start Recording", key="mic_button"):
            try:
                with st.spinner("Recording..."):
                    audio_file = st.session_state.audio_handler.record_audio()
                    if audio_file:
                        st.success("Recording completed!")
                        self.process_audio(audio_file)
                    else:
                        st.error("Failed to record audio. Please check your microphone.")
            except Exception as e:
                logger.error(f"Error recording audio: {e}")
                st.error("Error recording audio. Please try again.")

    def handle_file_upload(self) -> None:
        """Handle audio file upload interactions"""
        uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name
                
                self.process_audio(temp_path)
                
                # Cleanup
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Error processing uploaded file: {e}")
                st.error("Error processing uploaded file. Please try again.")

    def process_audio(self, audio_file: str) -> None:
        """Process audio and generate response"""
        try:
            result = st.session_state.pipeline.process_audio_query(
                st.session_state.audio_handler,
                audio_file,
                st.session_state.intents_data
            )
            
            if result:
                st.write("Your question:", result['query'])
                st.write("Response:", result['text_response'])
                st.audio(result['audio_response'])
            else:
                st.error("Could not process audio. Please try again.")
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            st.error("Error generating response. Please try again.")

    def run(self) -> None:
        """Run the application"""
        try:
            # Create tabs
            text_tab, mic_tab, upload_tab = st.tabs([
                "Text Input", 
                "Microphone Input", 
                "Upload Audio"
            ])

            # Handle different interaction methods
            with text_tab:
                self.handle_text_input()
                
            with mic_tab:
                self.handle_mic_input()
                
            with upload_tab:
                self.handle_file_upload()

            # Add disclaimer
            st.markdown("---")
            st.markdown("""
            **Disclaimer**: This medical assistant is for informational purposes only. 
            Always consult healthcare professionals for medical advice, diagnosis, or treatment.
            """)
            
        except Exception as e:
            logger.error(f"Error running application: {e}")
            st.error("An unexpected error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    app = MedicalAssistantApp()
    app.run()
