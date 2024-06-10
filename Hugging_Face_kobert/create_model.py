# ★ Hugging Face를 통한 모델 및 토크나이저 Import
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

def create_model(save_path='./kobert_model'):
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    
    # 모델 및 토크나이저 저장
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"Model and tokenizer saved to {save_path}")

if __name__ == "__main__":
    create_model()


### jhgan/ko-sroberta-sts 모델
# from transformers import AutoModel, AutoTokenizer

# model_name = "jhgan/ko-sroberta-sts"
# local_model_path = "./local_model"

# # 모델과 토크나이저 다운로드 및 저장
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model.save_pretrained(local_model_path)
# tokenizer.save_pretrained(local_model_path)