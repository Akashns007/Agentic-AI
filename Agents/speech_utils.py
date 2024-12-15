import sounddevice as sd
import speech_recognition as sr
from TTS.api import TTS
import soundfile as sf

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/fast_pitch", gpu=True)

WAKE_WORD = "nexa"
import re

def preprocess_text_for_speech(text):
    """
    Preprocess the text to skip URLs during TTS playback.
    """
    # Regular expression to match URLs
    url_pattern = r'https?://\S+|www\.\S+'
    # Replace URLs with a blank space to skip them
    cleaned_text = re.sub(url_pattern, "Link", text)
    return cleaned_text

def speak(text):
    """
    Converts the given text to speech, skipping URLs during playback.
    """
    # Preprocess the text to skip URLs
    try :
        cleaned_text = preprocess_text_for_speech(text)
        audio = tts.tts(cleaned_text, return_type="numpy")
        sd.play(audio, samplerate=22050)
        sd.wait()  # Wait until playback finishes
    except Exception as e:
        print(f"Error during speech playback: {e}")


def detect_wake_word():
    """
    Continuously listens for the wake word and then listens for the user's query once detected.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for the wake word...")
        while True:
            try:
                # Listen for the wake word
                wake_word_audio = recognizer.listen(source, timeout=10)
                command = recognizer.recognize_google(wake_word_audio).lower()
                if WAKE_WORD in command:
                    
                    data, samplerate = sf.read("example.wav")

                    # Play the audio
                    sd.play(data, samplerate)
                    sd.wait() 
                    
                    print("Wake word detected! Listening for your query...")

                    # Listen for the query
                    query_audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)
                    text = recognizer.recognize_google(query_audio)
                    print("You said: " + text)
                    return text  # Return the query
                
            except sr.UnknownValueError:
                print("Could not understand audio. Waiting for the wake word again...")
                continue
            except sr.RequestError as e:
                print(f"Error with speech recognition service: {e}")
                continue

if __name__ == "__main__":
    query = detect_wake_word()
    speak(query)