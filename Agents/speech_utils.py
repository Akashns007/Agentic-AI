import sounddevice as sd
import speech_recognition as sr
from TTS.api import TTS

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

WAKE_WORD = "nexa"

def speak(text):
    """
    Converts the given text to speech.
    """
    audio = tts.tts(text, return_type="numpy")
    sd.play(audio, samplerate=22050)
    sd.wait()  # Wait until playback finishes

def detect_wake_word():
    """
    Continuously listens for the wake word and then listens for the user's query once detected.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for the wake word...")
        while True:
            try:
                # Listen for the wake word with a timeout
                wake_word_audio = recognizer.listen(source, timeout=10)
                command = recognizer.recognize_google(wake_word_audio).lower()
                if WAKE_WORD in command:
                    print("Wake word detected! Listening for your query...")
                    
                    # Adjust settings for the query
                    recognizer.pause_threshold = 2.0
                    recognizer.dynamic_energy_threshold = True
                    
                    # Listen for the query
                    query_audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
                    text = recognizer.recognize_google(query_audio)
                    print("You said: " + text)
                    return text
            except sr.UnknownValueError:
                print("Could not understand audio. Waiting for the wake word again...")
                continue
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                continue

