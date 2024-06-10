import pandas as pd
import os

data_dir = r'C:\KorNLUDatasets\KorSTS'
train_file = os.path.join(data_dir, 'sts-train.tsv')
dev_file = os.path.join(data_dir, 'sts-dev.tsv')
test_file = os.path.join(data_dir, 'sts-test.tsv')

df = pd.read_csv(train_file, sep='\t', on_bad_lines='skip')
x = df[['sentence1','score', 'sentence2']]
# Display the DataFrame to check if it loaded correctly
print(x)
