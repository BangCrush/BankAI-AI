import torch
from transformers import BertModel, BertTokenizer, AdamW
import torch.nn.functional as F
from tqdm import tqdm
from train_model import train_model
import pandas as pd

# 학습된 모델과 토크나이저 불러오기
model_path = './trained_model'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


# CSV 파일에서 학습 데이터를 읽어오는 함수
def load_train_data(csv_file):
    df = pd.read_csv(csv_file)
    train_data = [(row['word1'], row['word2'], row['label']) for _, row in df.iterrows()]
    return train_data

# 추가 학습을 위한 데이터 준비
csv_file = r'C:\dev\BankAI-AI\Similarity\전처리\train_data.csv'  # CSV 파일 경로
additional_train_data = load_train_data(csv_file)


# 추가 학습
train_model(additional_train_data, model, tokenizer)
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
