from gtts import gTTS
import os
from playsound import playsound
def text2speech(text):
    tts = gTTS(text=text, tld='com.vn', lang='en')
    tts.save("hello.mp3")
    playsound("hello.mp3")
if __name__ == "__main__":
    text = "Hello"
    text2speech(text)