import torch
from transformers import BertModel, BertTokenizer, AdamW
import torch.nn.functional as F
from tqdm import tqdm

# BERT 모델과 토크나이저 로드
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# 학습을 위한 데이터 준비
train_data = [
    ("확인", "확인", 1), ("확인", "취소", 0), ("확인", "계좌이체", 0),
    ("취소", "확인", 0), ("취소", "취소", 1), ("취소", "계좌이체", 0),
    ("계좌이체", "확인", 0), ("계좌이체", "취소", 0), ("계좌이체", "계좌이체", 1),
    ("오케이","확인",1),("하지마","취소",1),("하지마","확인",0),("하지마","계좌이체",1)
]

def train_model(train_data, model, tokenizer, epochs=3):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for word1, word2, label in train_data:
            inputs1 = tokenizer(word1, return_tensors='pt')
            inputs2 = tokenizer(word2, return_tensors='pt')
            
            outputs1 = model(**inputs1).last_hidden_state[:, 0, :]
            outputs2 = model(**inputs2).last_hidden_state[:, 0, :]

            similarity = F.cosine_similarity(outputs1, outputs2).flatten()
            similarity = torch.sigmoid(similarity)  # 코사인 유사도를 0과 1 사이로 변환
            loss = F.binary_cross_entropy(similarity, torch.tensor([label], dtype=torch.float))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}")

# 모델 학습
train_model(train_data, model, tokenizer)

# 학습된 모델 저장
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
