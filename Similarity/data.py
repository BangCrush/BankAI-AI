import pandas as pd
import random

# 단어 목록 생성
words = ["확인", "취소", "계좌이체", "입금", "출금", "송금", "잔액조회", "예금", "대출", "이자"]

# 데이터 생성
data = {
    "단어1": [],
    "단어2": [],
    "유사도": []
}

# 임의의 데이터 100개 생성
random.seed(42)  # 재현성을 위해 시드 설정

for _ in range(100):
    word1 = random.choice(words)
    word2 = random.choice(words)
    similarity = 1 if word1 == word2 else 0
    data["단어1"].append(word1)
    data["단어2"].append(word2)
    data["유사도"].append(similarity)

# 데이터 프레임 생성
df = pd.DataFrame(data)

# 엑셀 파일로 저장
excel_file = "train_data_100.xlsx"
df.to_excel(excel_file, index=False)

print(f"데이터가 {excel_file} 파일로 저장되었습니다.")
