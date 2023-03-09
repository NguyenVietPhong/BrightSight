#import library
import speech_recognition as sr
import time
# Initialize recognizer class (for recognizing the speech)

def speech2text(filename):
    r = sr.Recognizer()

    # Reading Audio file as source
    # listening the audio file and store in audio_text variable

    with sr.AudioFile(filename) as source:
        
        audio_text = r.listen(source)
        
    # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:
            
            # using google speech recognition
            #Adding hindi langauge option
            text = r.recognize_google(audio_text, language = "vi-vn")
            return text
        
        except:
            print('Sorry.. run again...')
            return False
if __name__ == "__main__":
    filename = '1.wav'
    t0 = time.time()
    text = speech2text(filename)
    print(f"time: {time.time()-t0}",)
    print(text)
