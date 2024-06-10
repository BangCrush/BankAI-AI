from Hugging_Face_kobert.embedding_model import EmbeddingModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cos_sim(a, b):
    return cosine_similarity(a, b)

def find_most_similar_word(input_word, reference_words):
    model = EmbeddingModel()
    # 참조 문서의 임베딩 계산
    reference_embeddings = model.embed_documents(reference_words)
    
    # 입력 문서의 임베딩 계산
    input_embedding = np.array(model.embed_query(input_word)).reshape(1, -1)
    
    # 코사인 유사도 계산
    similarities = cos_sim(input_embedding, reference_embeddings)[0]
    
    # 각 참조 단어와의 유사도 출력
    for word, similarity in zip(reference_words, similarities):
        print(f"'{input_word}'와 '{word}'의 유사도: {similarity:.4f}")
    
    # 가장 유사한 단어 찾기
    best_match_index = np.argmax(similarities)
    return reference_words[best_match_index]

# 사용 예시
if __name__ == "__main__":
    input_word = "상품메인페이지 보여줘"
    reference_words = ["상품메인페이지", "전체계좌페이지", "거래내역페이지", "계좌이체페이지", "내정보페이지", "확인", "취소"]
    most_similar_word, similarity = find_most_similar_word(input_word, reference_words)
    print(f"가장 유사한 단어: {most_similar_word} (유사도: {similarity:.4f})")
