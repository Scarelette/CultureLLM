import jsonlines, time
import fire, os
import google.generativeai as genai
import urllib.request
import json, random
import ssl
from llm_response import getResponse

context_dict = dict()
with open("data/culture_context.jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        for key in item.keys():
            context_dict[key] = item[key]
    
def getFilePath(path):
    i = 0
    while os.path.exists(f"{path}.jsonl"):
        i += 1
        path = path + '_' + str(i)
    return path

def run(model_text, context):
    # Add more models for test
    model_dict = {'chatgpt': 'gpt-3.5-turbo'}
    if model_text in model_dict.keys():
        model = model_dict[model_text]
    else:
        model = model_text
    # num = 0
    
    path = getFilePath(f'data/China/CValues/output_context_{model_text}')

    with jsonlines.open(f'{path}.jsonl',mode='a') as writer:
        with open("data/China/CValues/cvalues_responsibility_mc.jsonl", "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):

                response = getResponse(item['prompt'], model, 'Chinese', context)
                item['response'] = response.strip()
                writer.write(item)
    print('ok')


if __name__ == '__main__':
    fire.Fire(run)    