import csv
import json

csv_file = 'data.csv'
jsonl_file = 'data.jsonl'

def process_csv_to_jsonl(csv_file, jsonl_file):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    jsonl_data = []
    for row in rows:
        if row['stereo_antistereo'] == 'stereo':
            data = row['sent_more']
            label = 'BAD'
        elif row['stereo_antistereo'] == 'antistereo':
            data = row['sent_less']
            label = 'GOOD'
        else:
            continue

        json_data = {'data': data, 'label': label}
        jsonl_data.append(json.dumps(json_data))

    with open(jsonl_file, 'w') as file:
        file.write('\n'.join(jsonl_data))

    print(f'Successfully converted {csv_file} to {jsonl_file}.')

process_csv_to_jsonl(csv_file, jsonl_file)