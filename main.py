import fire
import os
import random
import torch
# from llm_response import getResponse
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import codecs
import csv, jsonlines
import re
import time
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def getResponse(prompt, model_text):
    msg = [{"role": "user", "content": prompt}]
    # print('Msg: ', msg)
    client = OpenAI(api_key="xxx")

    output = None
    times = 0
    while output is None and times <= 10:
        try:
            times += 1  
            response = client.chat.completions.create(
                model=model_text,
                messages=msg,
                temperature=0.7
                )
            output = response.choices[0].message.content
        except Exception as e:
            print(e)
            print('Retrying...')
            time.sleep(5)
    if times >= 10:
        print('Failed! Model Input: ', prompt)
        output = ''

    return output

def getPrompt(s, n):
    prompt = ("Could you generate " + str(n) 
                + " sentences that (1) of different sentence structures and (2) of the same meaning with the following sentence: "
                + s # need n
                + ". Please number the generated sentences from 1 to " + str(n) + "."
                )
    return prompt

def getSynonymsPrompt(word, w_class):
    prompt = (f"Please generate 5 Synonyms for the word: {word}."
                + f"This is {w_class}" # need n
                + ". Please number the generated sentences from 1 to 5."
                )
    return prompt

def postProcess(s, model):
    if model == 'gpt4':
        s_list = s.split('\n')
        new_s_list = []
        for item in s_list:
            index = item.find('.')
            item = item[index+1:].strip()
            new_s_list.append(item)
    elif model == 'llama2':
        s_list = s.split('\n')[1:]
        new_s_list = []
        for item in s_list:
            nums = re.findall(r"\d+", item)
            if len(nums) > 0:
                index = item.find('.')
                item = item[index+1:].strip()
                new_s_list.append(item)
            else:
                new_s_list.append(item)

    return new_s_list

def calculate_similarity(sentence1, sentence2):
    def get_sentence_embedding(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)
        return sentence_embedding
    
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity

def sentence_filter(seed_s, sent_new, T=0.80):
    
    if calculate_similarity(seed_s, sent_new) > T:
        return 1
    return 0

def run(n=5, m=10):
    num = 0
    with open("data/WVQ.jsonl", "r+", encoding="utf8") as f:
        for row in jsonlines.Reader(f):
            ori_content = row['q_content']
            s = row['q_content']
            print('Num: ', num)

            num += 1
            print(s)

            # Step 1: generate sentences
            times = 0
            filtered_s_list = []
            new_prompt = getPrompt(s, n)
            while len(filtered_s_list) < n and times < 10:
                times += 1
                # output = get_response_from_llm('gpt4', [new_prompt])
                output = getResponse(new_prompt, 'gpt-4')
                # print('Output: ', output)
                s_list = postProcess(output, 'gpt4')
                print(s_list)
                for item in s_list:
                    if sentence_filter(s, item) == 1:
                        filtered_s_list.append(item)

            filtered_s_list = filtered_s_list[:n]
            print(len(filtered_s_list))
            print('Filtered S List: ', filtered_s_list)

            # Step 2: replace with synonyms
            final_sentences = []
            if len(filtered_s_list) == 0:
                filtered_s_list = [s]
            cur_len = int(m / len(filtered_s_list))
            print('Cur len: ', cur_len)
            for sentence in filtered_s_list:
                cur_list = []
                ori_words = []
                w_synonyms = []
                for sent in nltk.sent_tokenize(sentence):
                    # NER
                    # for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
                    #     if hasattr(chunk, 'label'):
                    #         print(chunk.label(), ' '.join(c[0] for c in chunk))
                    # noun
                    for word in nltk.pos_tag(nltk.word_tokenize(sent)):
                        if 'NN' == word[1] or 'NNS' == word[1] or 'NNP' == word[1]:
                            print(word[0])
                            ori_words.append(word)
                    # for word in nltk.pos_tag(nltk.word_tokenize(sent)):
                    #     if 'PRP' == word[1] or 'PRP$' == word[1] or 'NNP' == word[1]:
                    #         print(word[0])
                    #         ori_words.append(word)
                    # adj
                    for word in nltk.pos_tag(nltk.word_tokenize(sent)):
                        if 'JJ' == word[1]:
                            ori_words.append(word)
                    # adv
                    for word in nltk.pos_tag(nltk.word_tokenize(sent)):
                        if 'RB' == word[1]:
                            print(word[0])
                            ori_words.append(word)
                    # verb
                    for word in nltk.pos_tag(nltk.word_tokenize(sent)):
                        if 'VB' == word[1] or 'VBD' == word[1] or 'VBG' == word[1] or 'VBN' == word[1]:
                            print(word[0])
                            ori_words.append(word)
                for i in range(10):
                    if i >= 5 and len(cur_list) > cur_len:
                        break
                    new_sentence = sentence
                    # print('Ori word: ', ori_words)
                    for j in range(len(ori_words)):
                        word = ori_words[j]
                        if len(w_synonyms) < j + 1:
                            prompt = getSynonymsPrompt(word[0], word[1])
                            # output = get_response_from_llm('gpt4', [prompt])
                            output = getResponse(prompt, 'gpt-4')
                            # print('Sys: ', output)
                            synonyms = postProcess(output, 'gpt4')
                            if len(synonyms) > 0 and '.' not in synonyms[0]:
                                print('Synonyms: ', synonyms)
                                w_synonyms.append(synonyms)
                            else:
                                continue
                        if len(w_synonyms) > j:
                            synonyms = w_synonyms[j]
                            i = random.randint(0, len(synonyms) + 1)
                            if i < len(synonyms):
                                new_sentence = new_sentence.replace(word[0], synonyms[i].lower())
                                if sentence_filter(sentence, new_sentence) == 1:
                                    cur_list.append(new_sentence)
                
                cur_list = list(set(cur_list))
                sim = [calculate_similarity(s, new_s) for new_s in cur_list]
                cur_list_f = [x for _, x in sorted(zip(sim, cur_list), reverse=True)][: cur_len]
                # cur_list_f = [x for _, x in sorted(zip(sim, cur_list))][: cur_len]
                final_sentences.extend(cur_list_f)                       
            print(len(final_sentences))
            # final_sentences = filtered_s_list
            print(final_sentences)
            item = row
            item['ori_content'] = ori_content

            path = 'data/new_WVQ_test.jsonl'
            with jsonlines.open(path, mode='a') as writer:
                for new_s in final_sentences:
                    item['q_content'] = new_s
                    writer.write(item)

if __name__ == '__main__':
    fire.Fire(run)
