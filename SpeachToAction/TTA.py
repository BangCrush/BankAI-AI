#Text to Action module

def tta(text):
    if "뒤로" in text or "취소" in text:
        return {"action": "back"}
    elif "확인" in text:
        return {"action": "confirm"}
    else:
        return "I am not sure what you want me to do"