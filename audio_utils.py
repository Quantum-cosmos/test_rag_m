import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from gtts import gTTS
import pyttsx3
import os
from datetime import datetime
import numpy as np

class AudioHandler:
    def __init__(self, device=None):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

        try:
            self.device = device if device is not None else self.get_default_input_device()
        except OSError as e:
            print(f"Error: {e}. PortAudio may be missing. Audio features will be disabled.")
            self.device = None  # Disable audio recording

    def get_default_input_device(self):
        """Fetch and set the default input device"""
        try:
            devices = sd.query_devices()
            default_device = sd.default.device[0]  # Default input device
            print(f"Using default input device: {default_device}")
            return default_device
        except Exception as e:
            print(f"Error getting default input device: {e}")
            return None


    def record_audio(self, duration=5, sample_rate=44100):
        """Record audio from the microphone"""
        if self.device is None:
            print("No valid input device found!")
            return None

        print("Recording...")
        try:
            recording = sd.rec(int(duration * sample_rate),
                               samplerate=sample_rate,
                               channels=1,
                               dtype='float32',
                               device=self.device)
            sd.wait()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/input_{timestamp}.wav"
            os.makedirs("recordings", exist_ok=True)
            sf.write(filename, recording, sample_rate)
            return filename
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None

    def transcribe_audio(self, audio_file):
        """Transcribe an audio file to text"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

    def text_to_speech(self, text):
        """Convert text to speech and save as an audio file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"responses/response_{timestamp}.mp3"
        os.makedirs("responses", exist_ok=True)
        
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            return filename
        except Exception as e:
            print(f"Error converting text to speech: {e}")
            return None

    def play_audio(self, audio_file):
        """Play an audio file"""
        try:
            data, samplerate = sf.read(audio_file)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")
