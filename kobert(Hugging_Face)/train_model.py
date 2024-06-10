import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from tqdm import tqdm
from kobert_tokenizer import KoBERTTokenizer
import pandas as pd

class CustomPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        input_text = row['word1']
        reference_text = row['word2']
        label = row['label']
        inputs = self.tokenizer.encode_plus(
            input_text,
            reference_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].squeeze(0)  # 올바른 차원 조정을 위해 squeeze 사용
        attention_mask = inputs['attention_mask'].squeeze(0)  # 올바른 차원 조정을 위해 squeeze 사용
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

def pad_tensor(tensor, max_len):
    padded_tensor = torch.zeros(max_len, dtype=tensor.dtype)
    padded_tensor[:tensor.size(0)] = tensor
    return padded_tensor

def collate_fn(batch):
    max_len = max(item['input_ids'].size(0) for item in batch)
    input_ids = torch.stack([pad_tensor(item['input_ids'], max_len) for item in batch])
    attention_masks = torch.stack([pad_tensor(item['attention_mask'], max_len) for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'label': labels
    }

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model(dataloader, model, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]  # 변경된 부분: outputs[0]로 손실 추출
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_loss / len(dataloader)

def validate_model(dataloader, model, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]  # 변경된 부분: outputs[0]로 손실 추출
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # 사용자 정의 데이터셋 로드
    df = pd.read_csv(r'C:\dev\BankAI-AI\Similarity\전처리\train_data.csv')  # 새로운 데이터
    df = df.dropna()

    # 하이퍼파라미터 설정
    model_name = './kobert_model'  # 기존 모델 경로
    batch_size = 8
    epochs = 3
    max_len = 128
    learning_rate = 2e-5

    # 토크나이저 및 데이터로더 준비
    tokenizer = KoBERTTokenizer.from_pretrained(model_name)
    dataset = CustomPairDataset(df, tokenizer, max_len)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Dropout 확률 설정을 포함한 모델 구성 로드
    config = BertConfig.from_pretrained(model_name, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 학습 실행
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    for epoch in range(epochs):
        avg_loss = train_model(dataloader, model, optimizer, scheduler, device)
        val_loss = validate_model(dataloader, model, device)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_loss}, Validation Loss: {val_loss}')
        
        early_stopping(val_loss)
        if (early_stopping.early_stop):
            print("Early stopping")
            break

    # 모델 저장
    model.save_pretrained('./kobert_model')
    tokenizer.save_pretrained('./kobert_model')
    print(f"Updated model and tokenizer saved to ./kobert_model")

if __name__ == "__main__":
    main()
