number_dic_ko = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
place_value1_ko = ("", "십", "백", "천")
place_value2_ko = ("","만", "억", "조", "경")

def split_number(number, n):
    """number를 n자리씩 끊어서 리스트로 반환한다."""
    res = []
    div = 10**n
    while number > 0:
        number, remainder = divmod(number, div)
        res.append(remainder)
    return res


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

def number_to_word_ko(number, delimiter=" "):
    """0 이상의 number를 한글로 바꾼다.
       delimiter를 ''로 지정하면 1을 '일'로 바꾸고 공백을 넣지 않는다."""
    if number == 0:
        return "영"
    word_list = []
    for place, digits in enumerate(split_number(number, 4)):
        if word := convert_lt_10000(digits, delimiter):
            word += place_value2_ko[place]
            word_list.append(word)
    res = delimiter.join(word_list[::-1])   
    if delimiter and 10000 <= number < 20000:
        res = res[1:]
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
            sum_num = sum(number)
            if sum_num == 0:
                sum_num = 1
            res += sum_num * 10000 ** place_value2_ko.index(ch)
            number = []
    res += sum(number)
    return res


def logic(string):
    spl = string.split(' ')
    string = ''
    for i in spl:
        if i.isdigit():
            string+=number_to_word_ko(int(i))
        else:
            string+=i
            
    answer = ''
    digit = ''
    for s in string:
        if s.isdigit():
            digit += s
        else:
            if digit:
                answer += number_to_word_ko(int(digit))
                digit = ''
            answer += s
    if digit:
        answer += number_to_word_ko(int(digit))
    return word_to_number_ko(answer)


