import csv
import json

def tsv_to_jsonl(tsv_file, jsonl_file):
    with open(tsv_file, 'r', encoding='utf-8') as tsvfile, open(jsonl_file, 'w', encoding='utf-8') as jsonlfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            data = row['tweet']
            label = row['subtask_a']
            json_data = {'data': data, 'label': label}
            jsonlfile.write(json.dumps(json_data) + '\n')

tsv_file = 'data.tsv'
jsonl_file = 'data.jsonl'
tsv_to_jsonl(tsv_file, jsonl_file)