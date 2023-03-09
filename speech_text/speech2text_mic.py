import speech_recognition as sr
import time
import pyaudio

def speech2text_mic():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        # print('Say Something:')
        audio = r.listen(source)
        try:
            # using google speech recognition
            #Adding hindi langauge option
            text = r.recognize_google(audio, language = 'en-En')
            return text

            
        except:
            print("Sorry.. run again...")
            speech2text_mic()


if __name__ == "__main__":
    t0 = time.time()
    text = speech2text_mic() 
    # print(f"time: {time.time()-t0}")  
    print(text)