import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy

# 학습된 모델과 토크나이저 로드
model_name = './trained_model'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 기준이 되는  단어

def get_bert_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    outputs = model(**inputs)
    # BERT의 마지막 은닉층의 첫 번째 토큰 ([CLS]) 벡터를 사용
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

def find_similar_word(input_word, reference_words):
    # 기준 단어들과 입력 단어의 BERT 임베딩 계산
    pure_reference_words =  reference_words.copy()
    print(pure_reference_words)
    
    input_word = input_word.replace(" ", "")
    reference_words = [word.replace(" ", "") for word in reference_words]
    
    reference_embeddings = np.vstack([get_bert_embedding(word) for word in reference_words])
    input_embedding = get_bert_embedding(input_word)

    # 입력 단어 임베딩과 각 기준 단어 임베딩 간의 코사인 유사도 계산
    similarities = cosine_similarity(input_embedding, reference_embeddings).flatten()

    # 유사도가 가장 높은 기준 단어를 선택
    most_similar_index = similarities.argmax()

    # 각 단어와의 유사도 출력
    for word, similarity in zip(reference_words, similarities):
        print(f"입력값 '{input_word}'는 '{word}'와의 유사도: {similarity:.4f}")

    return pure_reference_words[most_similar_index]
