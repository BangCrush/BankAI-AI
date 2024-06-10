import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

class STSDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data.iloc[idx, 0]
        sentence2 = self.data.iloc[idx, 1]
        score = self.data.iloc[idx, 2]
        
        inputs = self.tokenizer.encode_plus(
            str(sentence1), 
            str(sentence2),
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'token_type_ids': inputs['token_type_ids'].squeeze(),
            'score': torch.tensor(score, dtype=torch.float)
        }

# 전처리된 데이터 로드
train_data = pd.read_csv('processed_train_data.csv')
dev_data = pd.read_csv('processed_dev_data.csv')

# 데이터셋 및 데이터로더 생성
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
train_dataset = STSDataset(train_data, tokenizer)
dev_dataset = STSDataset(dev_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=16)

# KoBERT 모델 초기화
model = BertForSequenceClassification.from_pretrained('monologg/kobert', num_labels=1)

# 학습 설정
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 손실 함수 설정
loss_fn = torch.nn.MSELoss()

def train_epoch(model, data_loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        scores = batch['score'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        logits = outputs.logits
        loss = loss_fn(logits.squeeze(), scores)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

# 모델 학습
epochs = 3
for epoch in tqdm(range(epochs)):
    train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}')

# 모델 저장 경로 설정
save_directory = 'trained_model'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 모델 저장
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
