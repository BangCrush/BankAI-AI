from similarity_model import find_similar_word

# 테스트
input_word = "오케이"
reference_words = ["상품 메인 페이지","전체 계좌 페이지","거래내역 조회 페이지","내 정보 페이지","계좌 이체 페이지","확인","취소"]
result = find_similar_word(input_word,reference_words)

print(f"입력값 '{input_word}'는 '{result}'와 더 유사합니다.")