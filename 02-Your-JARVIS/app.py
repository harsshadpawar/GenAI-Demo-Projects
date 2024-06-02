import streamlit as st
import os
import openai
import whisper
from gtts import gTTS
from openai import OpenAI
from pydub import AudioSegment
from PIL import Image
from pydub.playback import play
import pyaudio
import wave

# Your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Initialize the Whisper model
model = whisper.load_model("base")

# Function to record audio from the microphone
def record_audio(filename, duration, fs=44100, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=fs,
                    input=True,
                    frames_per_buffer=1024)

    frames = []
    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wave_file.setframerate(fs)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

# Function to convert speech to text using Whisper
def speech_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# Function to get response from GPT-4
def get_gpt4_response(prompt):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, respond in short to the user questions."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Function to convert text to speech
def text_to_speech(text, output_path):
    tts = gTTS(text)
    tts.save(output_path)
    sound = AudioSegment.from_file(output_path)
    play(sound)

# Streamlit UI setup
with st.sidebar:
# Load your image (replace with your image path)
    
    loaded_image = Image.open("./images/Jarvis4.jpg") 
    st.sidebar.image(loaded_image)  # Adjust width as needed

    # Sidebar content
    st.sidebar.title("💬JARVIS:")

st.header("J.A.R.V.I.S. (Demo)")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    formatted_content = f"{msg['content']}"
    st.chat_message(msg["role"]).write(formatted_content)

# Path to your audio file
recorded_audio_path = "recorded_audio.wav"
# Path to save the response audio
response_audio_path = "response.mp3"

if prompt := st.button("Record"):
    record_duration = 10  # seconds
    record_audio(recorded_audio_path, record_duration)
    prompt = speech_to_text(recorded_audio_path)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = get_gpt4_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(f"{response}")
    
    text_to_speech(response, response_audio_path)

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response = get_gpt4_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(f"{response}")

    text_to_speech(response, response_audio_path)
