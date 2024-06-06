number_dic_ko = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
place_value1_ko = ("", "십", "백", "천")
place_value2_ko = ("", "만", "억", "조", "경")
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

print(word_to_number_ko("이십")) # 20
print(word_to_number_ko("천삼 십사만")) # 23
print(word_to_number_ko("사조 오천육백 이억이십만삼천사백오십육")) # 203456