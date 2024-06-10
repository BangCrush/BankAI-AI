# Speech to Text module
import speech_recognition as sr
import re
from SpeachToAction.test import logic
import azure.cognitiveservices.speech as speechsdk

number_dic_ko = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
place_value1_ko = ("", "십", "백", "천")
place_value2_ko = ("", "만", "억", "조", "경")

numeric_corrections = {
        "다시": "-", "대시": "-",
        "영": "0", "공": "0",
        "일": "1", "하나": "1",
        "이": "2", "둘": "2",
        "삼": "3", "셋": "3",
        "사": "4", "넷": "4",
        "오": "5", "다섯": "5",
        "육": "6", "여섯": "6",
        "칠": "7", "일곱": "7",
        "팔": "8", "여덟": "8",
        "구": "9", "아홉": "9"
    }

def split_number(number, n): # 이미 만든 함수이므로 생략
    ...

def convert_lt_10000(number, delimiter):
    """10000 미만의 수를 한글로 변환한다.
       delimiter가 ''이면 1을 무조건 '일'로 바꾼다."""
    res = ""
    for place, digit in enumerate(split_number(number, 1)):
        if not digit:
            continue
        if delimiter and digit == 1 and place != 0:
            num = ""
        else:
            num = number_dic_ko[digit]
        res = num + place_value1_ko[place] + res
    return res

def word_to_number_ko(word):
    """한글을 숫자로 바꾼다."""
    if word == "영":
        return 0
    res, number = 0, []
    for ch in word.replace(" ", ""):
        if ch in number_dic_ko:
            number.append(number_dic_ko.index(ch))
        elif ch in place_value1_ko:
            place_value = 10 ** place_value1_ko.index(ch)
            if number and number[-1] in range(1, 10):
                number[-1] *= place_value
            else:
                number.append(place_value)
        else:
            res += sum(number) * 10000 ** place_value2_ko.index(ch)
            number = []
    res += sum(number)
    return res

# 계좌번호(숫자)
def correct_numeric_transcription(transcription):
    
    for key, value in numeric_corrections.items():
        transcription = re.sub(key, value, transcription)

    return transcription

# 돈
def correct_money_transcription(transcription):
    corrections = {
        "만원": "0000",
        "천원": "000",
        "백원": "00",
        "십원": "0",
        "억원": "00000000"
    }

    for key, value in corrections.items():
        transcription = re.sub(key, value, transcription)

    # 천, 백, 십, 공백 처리
    return korean_to_number(transcription.replace("원", ""))
    
def korean_to_number(transcription):
    # 공백 처리
    money, flag = 0, 0
    for num in transcription.split():
        try:
            money += int(num)
        except ValueError:
            transcription = transcription.replace(" ", "")
            flag = 1
            break
    if flag == 0:
        transcription = str(money)

    # 천, 백, 십 처리
    result = ""
    num_dict = {'십': '0', '백': '00', '천': '000', '만': '0000', '억': '00000'}
    flag = False
    for idx, char in enumerate(reversed(transcription)):
        if char.isdigit() or char == '-':
            result = char + result
        elif char in num_dict and not flag and idx != len(transcription) - 1:
            result = num_dict[char] + result
            flag = True
        if idx == len(transcription) - 1 and char in num_dict:
            result = "1" + result

    return result

# Azure
def sttAzure(key, region, audio_file):
    STT_result = ""
    language_code = 'ko-KR'

    # STT with azure
    speech_config = speechsdk.SpeechConfig(subscription=key, region=region, speech_recognition_language=language_code)
    audio_config = speechsdk.AudioConfig(filename=audio_file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()

    # Check Result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        STT_result = format(result.text)
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    
    STT_result = STT_result.replace(".", "")
    STT_result = STT_result.replace(",", "")
    STT_result = STT_result.replace("?", "")
    # print(STT_result)
    
    string = STT_result.replace("원", "")
  

    STT_result = correct_numeric_transcription(STT_result)
    # print(STT_result)
    STT_result = correct_money_transcription(STT_result.replace(" 원", "원"))

    # print(STT_result)
    # return STT_result
    return logic(string)

# Azure 마이크
def from_mic(STT_result,type):
    # language_code = 'ko-KR'
    # speech_config = speechsdk.SpeechConfig(subscription=key, region=region, speech_recognition_language=language_code)
    # speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    # print("Speak into your microphone.")
    # STT_result = speech_recognizer.recognize_once_async().get()
    STT_result = STT_result.replace(".", "")
    STT_result = STT_result.replace(",", "")
    STT_result = STT_result.replace("?", "")
    print(STT_result)
    if type== "number":
        for c in numeric_corrections.keys():
            if c in STT_result:
                STT_result = correct_numeric_transcription(STT_result)
                STT_result = STT_result.replace(" ", "")
                STT_result = STT_result.replace("-", "")
                break
        if "원" in STT_result:
            STT_result =  logic(STT_result.split('원')[0])
    print(STT_result)
    return STT_result

# Google
def stt(audio_file):
    r = sr.Recognizer()
    harvard = sr.AudioFile(audio_file)
    with harvard as source:
        audio = r.record(source)
    result = r.recognize_google(audio ,language='ko-KR')
    corrected_result = correct_numeric_transcription(result).replace(" ", "")
    print(result)
    return corrected_result