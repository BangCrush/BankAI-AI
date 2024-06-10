import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_embedding(sentence, model, tokenizer, device):
    # 입력 문장 토크나이징
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    # 입력 데이터를 적절한 디바이스로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 모델이 올바른 디바이스에 있는지 확인
    model.to(device)
    
    # 효율성을 위해 그래디언트 계산 비활성화
    with torch.no_grad():
        outputs = model(**inputs)
    
    # [CLS] 토큰의 임베딩 추출
    last_hidden_state = outputs[0]  # outputs의 첫 번째 요소가 last_hidden_state
    embedding = last_hidden_state[:, 0, :].squeeze()
    
    return embedding

def find_similar_word(input_word, reference_words):
    model_name = './kobert_model'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 모델과 토크나이저 로드
    tokenizer = KoBERTTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    
    input_embedding = get_embedding(input_word, model, tokenizer, device).cpu().numpy()
    max_similarity = -1
    most_similar_word = None
    for word in reference_words:
        ref_embedding = get_embedding(word, model, tokenizer, device).cpu().numpy()
        similarity = cosine_similarity([input_embedding], [ref_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_word = word
        print(f"Input: {input_word}, Reference: {word}, Similarity: {similarity}")
    return most_similar_word, max_similarity
