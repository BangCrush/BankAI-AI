from similarity_model import find_similar_word

# 테스트
input_word = "질문 : 계좌이체 취소할까요 답 : 확인"
result = find_similar_word(input_word)
print(f"입력값 '{input_word}'는 '{result}'와 더 유사합니다.")
