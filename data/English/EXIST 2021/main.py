import csv
import json

def process_tsv_to_jsonl(input_file, output_file_en, output_file_es):
    with open(input_file, 'r', encoding='utf-8') as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter='\t')
        data_en = []
        data_es = []
        for row in reader:
            item = {
                'data': row['text'],
                'label': 'BAD' if row['task1'] == 'sexist' else 'GOOD'
            }
            if row['language'] == 'en':
                data_en.append(item)
            elif row['language'] == 'es':
                data_es.append(item)

    with open(output_file_en, 'w', encoding='utf-8') as en_file:
        for item in data_en:
            en_file.write(json.dumps(item) + '\n')

    with open(output_file_es, 'w', encoding='utf-8') as es_file:
        for item in data_es:
            es_file.write(json.dumps(item) + '\n')

# 处理 train.tsv 数据，并根据 language 属性分类数据到 en_data.jsonl 和 es_data.jsonl
process_tsv_to_jsonl('train.tsv', 'en_data.jsonl', 'es_data.jsonl')

# 处理 test.tsv 数据，并根据 language 属性分类数据到 en_data.jsonl 和 es_data.jsonl
process_tsv_to_jsonl('test.tsv', 'en_data.jsonl', 'es_data.jsonl')