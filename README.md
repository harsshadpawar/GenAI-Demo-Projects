# source C:/Users/harsh/anaconda3/Scripts/activate base
	conda env remove
	Deactive conda : conda deactivate

# conda activate devenv

# streamlit run app.py

# J.A.R.V.I.S. (Just A Rather Very Intelligent System) Demo

This project demonstrates a simple voice-activated assistant powered by OpenAI's Whisper for speech-to-text, GPT-4 for responses, and GTTS for text-to-speech conversion. It's a basic version inspired by the J.A.R.V.I.S. AI from Iron Man.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Harsshadpawar06/GenAI-Demo-Projects.git
   cd jarvis-demo
	 

2. **Install Dependencies:**
	```bash
	pip install streamlit openai whisper gtts pydub Pillow pyaudio wave 

3. **Set Your OpenAI API Key:**

Create a .env file in the project directory.
Add your OpenAI API key: OPENAI_API_KEY=<your-key>

Running the Demo
1. **Start the Streamlit App:** 
	```bash
		streamlit run app.py

2. **Interact with J.A.R.V.I.S.:** 

Click the "Record" button to initiate a 10-second voice recording.
J.A.R.V.I.S. will transcribe your speech, process it with GPT-4, and respond audibly.
Alternatively, type your query in the chat input box.

3. **How it Works**

1. Speech Input:

* Your voice is recorded and saved as a WAV file.
* The Whisper model transcribes the audio into text.

2. GPT-4 Processing:

* The transcribed text is sent to GPT-4 as a prompt.
* GPT-4 generates a concise and helpful response.

3. Text-to-Speech Output:

* The GTTS library converts the GPT-4 response into speech.
* The generated audio is played back to you.