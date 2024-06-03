# Speech to Text module
import speech_recognition as sr

def stt(audio_file):
    r = sr.Recognizer()
    harvard = sr.AudioFile(audio_file)
    with harvard as source:
        audio = r.record(source)
    result = r.recognize_google(audio ,language='ko-KR')
    print(result)
    return result
    