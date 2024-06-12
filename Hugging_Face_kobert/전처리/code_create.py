import pandas as pd

def add_data_to_csv(input_word, match_word, reference_words, csv_file):
    # 기존 CSV 파일 불러오기
    df = pd.read_csv(csv_file)

    # 새로운 행들을 저장할 리스트 생성
    new_rows = []

    # match_word에 대해 새로운 행 생성 (label 1)
    new_rows.append([input_word, match_word, 1])

    # 다른 참조 단어들에 대해 새로운 행 생성 (label 0)
    for ref_word in reference_words:
        if ref_word != match_word:
            new_rows.append([input_word, ref_word, 0])

    # 새로운 행들을 데이터프레임으로 변환
    new_df = pd.DataFrame(new_rows, columns=df.columns)

    # 기존 데이터프레임과 새로운 데이터프레임을 합치기
    updated_df = pd.concat([df, new_df], ignore_index=True)

    # 업데이트된 데이터프레임을 CSV 파일에 저장
    updated_df.to_csv(csv_file, index=False)
    return updated_df

# 입력 단어 및 참조 단어들
input_word = "이체" # 보이스로 입력되는 데이터 입력
match_word = "계좌이체페이지" # 실제 매칭시킬 단어 혹은 액션
reference_words = ["상품메인페이지", "전체계좌페이지", "거래내역조회페이지", "계좌이체페이지","내정보페이지", "확인", "취소"]

# CSV 파일 경로
csv_file = r'C:\dev\BankAI-AI\Similarity\전처리\train_data.csv'

# 데이터 추가 및 저장
updated_df = add_data_to_csv(input_word, match_word, reference_words, csv_file)

# 결과 출력
print(updated_df.tail(10))
