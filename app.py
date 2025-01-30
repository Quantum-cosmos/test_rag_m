import streamlit as st
import os

# Try to import audio dependencies, handle errors gracefully
try:
    import sounddevice as sd
    from audio_utils import AudioHandler
    audio_available = True
except OSError:
    audio_available = False
    st.warning("Audio recording is disabled because the required dependencies are missing.")

# Import MedicalRAGPipeline (Make sure this module exists)
from medical_rag import MedicalRAGPipeline

# 🔐 Load API Key securely (Avoid hardcoding it in the script)
API_KEY = os.getenv("GEMINI_API_KEY")  # Set this in your Streamlit Cloud secrets

def initialize_session_state():
    """Initializes all session state variables to avoid errors"""
    
    # Initialize audio handler
    if 'audio_handler' not in st.session_state:
        if audio_available:
            st.session_state.audio_handler = AudioHandler()
            if st.session_state.audio_handler.device is None:
                st.warning("Audio recording is disabled due to missing dependencies.")
        else:
            st.session_state.audio_handler = None

    # Initialize RAG pipeline
    if 'pipeline' not in st.session_state:
        if not API_KEY:
            st.error("❌ API Key is missing! Set GEMINI_API_KEY in Streamlit secrets.")
            return

        st.session_state.pipeline = MedicalRAGPipeline(gemini_api_key=API_KEY)
        
        # Load data for the model
        try:
            documents = st.session_state.pipeline.load_diseases_data("diseases.json")
            st.session_state.intents_data = st.session_state.pipeline.load_intents_data("intents.json")
            st.session_state.pipeline.create_index(documents)
        except FileNotFoundError:
            st.error("❌ Missing required data files (diseases.json, intents.json). Please upload them.")

    # Initialize other session state variables
    st.session_state.setdefault('messages', [])
    st.session_state.setdefault('audio_responses', {})

# Run session state initialization
initialize_session_state()

# 💬 Streamlit UI Components
st.title("🩺 Medical RAG Chatbot")

# Chat Input
user_input = st.text_input("Ask a medical question:", key="user_input")

# Handle User Query
if user_input:
    if 'pipeline' in st.session_state:
        response = st.session_state.pipeline.query(user_input)
        st.session_state.messages.append({"user": user_input, "bot": response})
        st.write(f"**Chatbot:** {response}")
    else:
        st.error("Medical RAG pipeline is not initialized.")

# Display Chat History
if st.session_state.messages:
    st.subheader("Chat History")
    for chat in st.session_state.messages:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")


def main():
    st.set_page_config(page_title="Medical Chat Assistant", page_icon="🩺", layout="wide")
    initialize_session_state()
    
    st.markdown("""
    <style>
    .chat-container {
        max-width: 800px;
        margin: auto;
    }
    .chat-message {
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 16px;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .user-message {
        background-color: #4CAF50;
        color: white;
        text-align: right;
        justify-content: flex-end;
    }
    .assistant-message {
        background-color: #2196F3;
        color: white;
        text-align: left;
        justify-content: flex-start;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🩺 Medical Chat Assistant")
    st.write("Ask medical questions through text, voice, or uploaded audio.")
    
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
        st.markdown(f'<div class="chat-container"><div class="chat-message {role_class}"><span class="avatar">{avatar}</span><strong>{message["role"]}:</strong> {message["content"]}</div></div>', unsafe_allow_html=True)
        
        if message["role"] == "assistant" and message["content"] in st.session_state.audio_responses:
            st.audio(st.session_state.audio_responses[message["content"]])
    
    text_input = st.text_input("Type your question:", key="text_query", on_change=lambda: st.session_state.__setitem__('text_query_submit', True))
    if st.session_state.get('text_query_submit', False) or st.button("Send"):
        if text_input:
            st.session_state.messages.append({"role": "user", "content": text_input})
            response = st.session_state.pipeline.generate_response(text_input, st.session_state.intents_data)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Generate and play audio response
            audio_file = st.session_state.audio_handler.text_to_speech(response)
            if audio_file:
                st.session_state.audio_responses[response] = audio_file
            
            st.session_state.text_query_submit = False
            st.rerun()
    
    st.subheader("🎙️ Voice Input")
    if st.button("🎤 Record Audio"):
        with st.spinner("Recording..."):
            audio_file = st.session_state.audio_handler.record_audio()
        if audio_file:
            result = st.session_state.pipeline.process_audio_query(st.session_state.audio_handler, audio_file, st.session_state.intents_data)
            if result:
                st.session_state.messages.append({"role": "user", "content": result['query']})
                st.session_state.messages.append({"role": "assistant", "content": result['text_response']})
                
                # Generate and play audio response
                st.session_state.audio_responses[result['text_response']] = result['audio_response']
                st.rerun()
    
    st.subheader("📂 Upload Audio")
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])
    if uploaded_file and st.button("Process Audio"):
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        result = st.session_state.pipeline.process_audio_query(st.session_state.audio_handler, temp_file, st.session_state.intents_data)
        os.remove(temp_file)
        if result:
            st.session_state.messages.append({"role": "user", "content": result['query']})
            st.session_state.messages.append({"role": "assistant", "content": result['text_response']})
            
            # Generate and play audio response
            st.session_state.audio_responses[result['text_response']] = result['audio_response']
            st.rerun()
    
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.audio_responses = {}
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    **Disclaimer**: This medical assistant is for informational purposes only. 
    Always consult healthcare professionals for medical advice, diagnosis, or treatment.
    """)

if __name__ == "__main__":
    main()
