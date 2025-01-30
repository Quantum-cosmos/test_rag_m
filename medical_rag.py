import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import logging
import google.generativeai as genai
from dataclasses import dataclass
import re
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    content: str
    metadata: Dict

class MedicalRAGPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', gemini_api_key: str = None):
        """
        Initialize the Medical RAG Pipeline
        Args:
            model_name: Name of the sentence transformer model
            gemini_api_key: API key for Gemini
        """
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        key="AIzaSyAulFzIkm9yvMawZBV5-HFoCEEu2BRzn7A"
        gemini_api_key=key
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        
        # Add exit patterns
        self.exit_patterns = {
            'quit', 'exit', 'bye', 'goodbye', 'terminate', 'end', 'sign off',
            'terminate the call', 'sign off', 'end call'
        }
        
        # Medical disclaimer
        self.medical_disclaimer = ("\nDisclaimer: This information is for educational purposes only. "
                                 "Please consult a healthcare professional for medical advice, diagnosis, or treatment.")

    def load_diseases_data(self, diseases_path: str) -> List[Document]:
        """
        Load and process diseases data from JSON file
        """
        try:
            with open(diseases_path, 'r') as f:
                data = json.load(f)
            
            documents = []
            for disease in data['diseases']:
                # Create comprehensive document for each disease
                content = f"{disease.get('tag', '')}: {disease.get(disease['tag'], '')}\n"
                content += f"Symptoms: {disease.get('symptoms', '')}\n"
                content += f"Treatment: {disease.get('treatment', '')}\n"
                if 'types' in disease:
                    content += f"Types: {disease['types']}\n"
                if 'prevention' in disease:
                    content += f"Prevention: {disease['prevention']}"
                
                doc = Document(
                    content=content,
                    metadata={'tag': disease['tag']}
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} disease documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading diseases data: {e}")
            raise

    def load_intents_data(self, intents_path: str) -> Dict:
        """
        Load intents data for response generation
        """
        try:
            with open(intents_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded intents data with {len(data['intents'])} intents")
            return data
        except Exception as e:
            logger.error(f"Error loading intents data: {e}")
            raise

    def create_index(self, documents: List[Document]):
        """
        Create FAISS index from documents
        """
        try:
            # Convert documents to embeddings
            texts = [doc.content for doc in documents]
            embeddings = self.encoder.encode(texts)
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(embeddings).astype('float32'))
            self.documents = documents
            
            logger.info(f"Created FAISS index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def normalize_query(self, query: str) -> str:
        """
        Normalize the query by removing special characters and converting to lowercase
        """
        # Remove special characters and convert to lowercase
        query = re.sub(r'[^\w\s]', '', query.lower())
        # Handle common misspellings
        misspellings = {
            'canser': 'cancer',
            'diabetis': 'diabetes',
            'artritis': 'arthritis',
            'highbloodpressure': 'hypertension',
            'asma': 'asthma',
            'hart': 'heart',
            'stroke': 'stroke',
            'alzheimers': 'alzheimer',
            'highblood': 'hypertension'
        }
        return misspellings.get(query, query)

    def is_exit_request(self, query: str) -> bool:
        """
        Check if the query is an exit request
        """
        normalized_query = query.lower().strip()
        return any(exit_pattern in normalized_query for exit_pattern in self.exit_patterns)

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """
        Search the index for similar documents
        """
        try:
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Search index
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), k
            )
            
            # Return documents and scores
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                results.append((self.documents[idx], float(distance)))
            
            return results
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise


    

    def generate_response(self, query: str, intents_data: Dict) -> str:
        """
        Generate improved response using RAG and intents data
        """
        try:
            # Check for exit request
            if self.is_exit_request(query):
                return "Thank you for using our medical assistance service. Remember to consult healthcare professionals for medical advice. Goodbye!"

            # Normalize query
            normalized_query = self.normalize_query(query)
            
            # Search for relevant documents
            search_results = self.search(normalized_query)
            
            # Create context from search results
            context = "\n\n".join([doc.content for doc, _ in search_results])
            
            # Find matching intent
            matching_intent = None
            for intent in intents_data['intents']:
                if any(pattern.lower() in normalized_query for pattern in intent['patterns']):
                    matching_intent = intent
                    break
            
            # Generate prompt with improved instructions
            prompt = f"""As a medical assistant, provide a clear and concise answer to the following question using the provided context. 
            Focus on accuracy and avoid repetition.

            Context:
            {context}

            Question: {query}

            Guidelines:
            1. Be concise and avoid repeating information
            2. If information is not available in the context, acknowledge that
            3. For symptoms or serious conditions, recommend consulting a healthcare provider
            4. Ensure the response is clear and well-structured
            5. If this is an emergency condition, emphasize seeking immediate medical attention
            """
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            final_response = response.text.strip()
            
            # If we have a matching intent, enhance with intent response
            if matching_intent:
                intent_response = matching_intent['responses'][0]
                # Combine responses while avoiding repetition
                if intent_response not in final_response:
                    final_response = f"{final_response}\n\n{intent_response}"
            
            # Add medical disclaimer
            final_response += self.medical_disclaimer
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ("I apologize, but I'm having trouble generating a response. "
                   "Please try again or consult a healthcare professional for medical advice.")


    def process_audio_query(self, audio_handler, audio_file, intents_data):
        """
        Process a query from an audio file
        """
        try:
            # Transcribe audio to text
            query = audio_handler.transcribe_audio(audio_file)
            if not query:
                return None, "Could not transcribe audio"

            # Generate text response
            text_response = self.generate_response(query, intents_data)

            # Convert response to audio
            audio_response = audio_handler.text_to_speech(text_response)

            return {
                'query': query,
                'text_response': text_response,
                'audio_response': audio_response
            }

        except Exception as e:
            logger.error(f"Error processing audio query: {e}")
            return None