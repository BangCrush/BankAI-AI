from similarity_calculator import find_most_similar_word

# 테스트
input_word = "내가 예금가입하고 싶어"
reference_words = ["상품메인페이지", "전체계좌페이지", "거래내역페이지", "계좌이체페이지", "내정보페이지", "확인", "취소","맞아"
                   , "예금 가입 페이지"]
result = find_most_similar_word(input_word, reference_words)
print(f"입력값 '{input_word}'는 '{result}'와 더 유사합니다")
