import os
import queue
import sounddevice as sd
import numpy as np
import json
import time
from vosk import Model, KaldiRecognizer
import pyttsx3
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 8000

# Check for Vosk model
if not os.path.exists("vosk-model-en"):
    print("Please download the Vosk model from:")
    print("https://alphacephei.com/vosk/models and unpack as 'vosk-model-en' in the current folder.")
    exit(1)

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Initialize Vosk model
print("Loading Vosk model...")
model = Model("try 3")
recognizer = KaldiRecognizer(model, SAMPLE_RATE)

# Initialize LangChain components
persist_directory = "doc_db"
embedding = HuggingFaceEmbeddings()
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
retriever = vectorstore.as_retriever()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def record_audio():
    """Record audio and return recognized text"""
    q = queue.Queue()
    device_info = sd.query_devices(None, 'input')
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))
    
    try:
        # Open audio stream
        with sd.RawInputStream(samplerate=SAMPLE_RATE,
                             blocksize=BLOCK_SIZE,
                             device=None,  # Use default device
                             dtype=np.int16,
                             channels=CHANNELS,
                             callback=callback):
            
            print("\nListening... Speak your question (Press Ctrl+C when done)")
            
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        return text
                
    except KeyboardInterrupt:
        print("\nStopped listening.")
        result = json.loads(recognizer.FinalResult())
        return result.get("text", "")
    except Exception as e:
        print(f"\nError recording audio: {e}")
        return ""

def speak_response(text):
    """Convert text to speech"""
    try:
        print("Speaking response...")
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def user_query(query):
    try:
        response = qa_chain.invoke({"query": query})
        return response["result"]
    except Exception as e:
        print(f"Error in query processing: {e}")
        return "I'm sorry, I encountered an error processing your query."

def main():
    print("\nVoice Assistant initialized. You can use text or voice input.")
    print("For voice input, speak clearly and press Ctrl+C when done speaking.")
    
    while True:
        try:
            input_type = input("\nDo you want to use text or voice input? (text/voice/quit): ").lower()
            
            if input_type == "quit":
                print("bye!")
                break
                
            if input_type == "text" or input_type == "Text" or input_type == "TEXT" or input_type == "text ":
                query = input("ASK YOUR QUESTION: ")
            elif input_type == "voice" or input_type == "Voice" or input_type == "VOICE" or input_type == "voice ":
                query = record_audio()
                if not query:
                    print("No speech detected. Please try again.")
                    continue
                print(f"\nRecognized text: {query}")
            else:
                print("Invalid input type. Please choose 'text' or 'voice' or 'quit'.")
                continue
            
            if query:
                answer = user_query(query)
                print("\nANSWER:", answer)
                speak_response(answer)
                print("\n-------------------------------------------")
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
