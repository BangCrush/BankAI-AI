import pandas as pd
import os

def load_korsts_data(file_path):
    # 데이터 파일 로드
    df = pd.read_csv(file_path, delimiter='\t', header=None, on_bad_lines='skip',skiprows=1)

    # 열 이름 지정
    df.columns = ['genre', 'filename', 'year', 'id', 'score', 'sentence1', 'sentence2']
    
    # NaN 값 처리 및 score 열을 float으로 변환
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    
    # 필요한 열 선택
    return df[['sentence1', 'sentence2', 'score']]


# 데이터 파일 경로 설정
data_dir = r'C:\KorNLUDatasets\KorSTS'
train_file = os.path.join(data_dir, 'sts-train.tsv')
dev_file = os.path.join(data_dir, 'sts-dev.tsv')
test_file = os.path.join(data_dir, 'sts-test.tsv')

# 데이터 로드
train_data = load_korsts_data(train_file)
dev_data = load_korsts_data(dev_file)
test_data = load_korsts_data(test_file)

# 데이터 크기 확인
print(f"Train data size: {train_data.shape[0]}")
print(f"Dev data size: {dev_data.shape[0]}")
print(f"Test data size: {test_data.shape[0]}")

# 전처리된 데이터를 저장합니다.
train_data.to_csv('processed_train_data.csv', index=False)
dev_data.to_csv('processed_dev_data.csv', index=False)
test_data.to_csv('processed_test_data.csv', index=False)

print(train_data.head())
