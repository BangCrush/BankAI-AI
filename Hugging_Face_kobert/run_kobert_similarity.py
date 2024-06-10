from use_model import find_similar_word

# 테스트
input_word = "송금할래"
reference_words = ["상품메인페이지", "전체계좌페이지", "거래내역페이지", "계좌이체페이지", "내정보페이지", "확인", "취소"]
result, similarity = find_similar_word(input_word, reference_words)
print(f"입력값 '{input_word}'는 '{result}'와 더 유사합니다. (유사도: {similarity})")

# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # 한국어로 잘 학습된 모델
# model_name = "jhgan/ko-sbert-nli"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True}

# hf = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

# def cos_sim(a, b):
#     return cosine_similarity(a, b)

# def find_most_similar_word(input_word, reference_words):
#     # 참조 문서의 임베딩 계산
#     reference_embeddings = hf.embed_documents(reference_words)
    
#     # 입력 문서의 임베딩 계산
#     input_embedding = np.array(hf.embed_query(input_word)).reshape(1, -1)
    
#     # 코사인 유사도 계산
#     similarities = cos_sim(input_embedding, reference_embeddings)[0]
    
#     # 각 참조 단어와의 유사도 출력
#     for word, similarity in zip(reference_words, similarities):
#         print(f"'{input_word}'와 '{word}'의 유사도: {similarity:.4f}")
    
#     # 가장 유사한 단어 찾기
#     best_match_index = np.argmax(similarities)
#     return reference_words[best_match_index], similarities

# # 사용 예시
# input_word = "이체할래"
# reference_words = ["상품메인페이지", "전체계좌페이지", "거래내역페이지", "계좌이체페이지", "내정보페이지", "확인", "취소"]

# most_similar_word, similarities = find_most_similar_word(input_word, reference_words)
# print(f"가장 유사한 단어: {most_similar_word}")
