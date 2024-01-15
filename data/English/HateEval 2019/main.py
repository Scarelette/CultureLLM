import pandas as pd
import jsonlines

# 定义输入文件名和输出文件名
input_files = ['hateval2019_en_dev.csv', 'hateval2019_en_test.csv', 'hateval2019_en_train.csv']
output_file = 'data.jsonl'

# 定义label的映射字典
label_mapping = {0: 'NOT', 1: 'OFF'}

# 创建一个空的DataFrame来存储所有数据
merged_data = pd.DataFrame(columns=['data', 'label'])

# 读取每个输入文件并进行处理
for file in input_files:
    # 读取CSV文件
    df = pd.read_csv(file)

    # 映射label列
    df['HS'] = df['HS'].map(label_mapping)

    # 选择需要的列并重命名
    df = df[['text', 'HS']]
    df.columns = ['data', 'label']

    # 将当前文件的数据添加到合并的DataFrame中
    merged_data = merged_data.append(df, ignore_index=True)

# 将合并的数据写入JSONL文件
with jsonlines.open(output_file, mode='w') as writer:
    for _, row in merged_data.iterrows():
        writer.write(row.to_dict())