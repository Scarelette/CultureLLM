import pandas as pd
import json

# 读取CSV文件
df = pd.read_csv('data.csv')

# 根据语言分组
grouped = df.groupby(df['cn_id'].str[:2])

# 定义转换函数
def transform_data(row):
    en_data = {'data': row['hateSpeech'], 'label': 'HS'}
    en_data = json.dumps(en_data)

    fr_data = {'data': row['counterSpeech'], 'label': 'NOT_HS'}
    fr_data = json.dumps(fr_data)

    return en_data, fr_data

# 处理数据并生成JSONL文件
for lang, group in grouped:
    if lang == 'EN':
        filename = 'en_data.jsonl'
    elif lang == 'FR':
        filename = 'fr_data.jsonl'
    elif lang == 'IT':
        filename = 'it_data.jsonl'
    else:
        continue

    group = group.drop_duplicates()  # 去重处理

    with open(filename, 'w') as f:
        for _, row in group.iterrows():
            en_data, fr_data = transform_data(row)
            f.write(en_data + '\n')
            f.write(fr_data + '\n')