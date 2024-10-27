import jsonlines
import re, random, os, fire
import codecs, csv

q_list = ['27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '122', '123', '124', '125', '126', '127', '128', '129', '132', '133', '134', '135', '136', '137', '138', '158', '159', '160', '161', '162', '169', '170', '196', '197', '198', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233']

def generateCountryAns(country, country_code):
    num = 0
    f_num = 0
    item_list = []
    avg_item = {'B_COUNTRY': 'Avg', 'B_COUNTRY_ALPHA': ''}
    avg_first_item = {'B_COUNTRY': 'Avg_First', 'B_COUNTRY_ALPHA': ''}
    avg_last_item = {'B_COUNTRY': 'Avg_Last', 'B_COUNTRY_ALPHA': ''}
    with codecs.open('data/WVS_Cross-National_Wave_7_csv_v5_0.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            if row['B_COUNTRY'] == country_code:
                num += 1
                if num <= 1000:
                    f_num += 1
                item = {'B_COUNTRY': row['B_COUNTRY'], 'B_COUNTRY_ALPHA': row['B_COUNTRY_ALPHA']}
                for q in q_list:
                    k = 'Q' + q
                    item[k] = row[k]
                    if num <= 1000:
                        if k not in avg_item.keys():
                            avg_item[k] = int(row[k])
                            avg_first_item[k] = int(row[k])
                        else:
                            avg_item[k] += int(row[k])
                            avg_first_item[k] += int(row[k])
                    else:
                        if k not in avg_last_item.keys():
                            avg_last_item[k] = int(row[k])
                            if k not in avg_item.keys():
                                avg_item[k] = int(row[k])
                        else:
                            avg_item[k] += int(row[k])
                            avg_last_item[k] += int(row[k])

                print(item)
                item_list.append(item)
    f.close()
    print('Num: ', num)

    print('First: ', avg_first_item)
    print('Last: ', avg_last_item)
    print('F Num: ', f_num)

    for q in q_list:
        k = 'Q' + q
        avg_item[k] /= len(item_list)
        avg_first_item[k] /= f_num
        avg_last_item[k] /= len(item_list) - f_num

    dir_path = f"data/{country}"
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

    with open(f'{dir_path}/{country}.csv', 'w', newline='') as f:
        data = item_list[0]
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        for item in item_list:
            writer.writerow(item)
        writer.writerow(avg_item)
        writer.writerow(avg_first_item)
        writer.writerow(avg_last_item)

def getPrompt(item, t, hasContext=False):
    from llm_response import get_response_from_llm
    content = item['q_content']
    option = item['option']
    nums = re.findall(r"\d+",option)

    # if t % 2 == 1:
    #     p_prompt = getPassivePrompt(content)
    #     content = get_response_from_llm('gpt4', [p_prompt])[0]

    if '?' in content:
        prompt = f"Give me the answer from {min(nums)} to {max(nums)}: {content} {option}. You can only choose one option."
    else:
        prompt = f"Give me the answer from {min(nums)} to {max(nums)}: Do you agree with {content}? {option}. You can only choose one option."
 
    # if hasContext == True:
    #     num = random.randint(0, len(contexts)-1)
    #     cur_context = contexts[num]
    #     prompt = cur_context + ' ' + prompt

    return prompt

def translate(content, language="Arabic"):
    from llm_response import get_response_from_llm
    prompt = f'Please translate to {language}: {content}'
    output = get_response_from_llm('gpt4', [prompt])[0]

    return output

def generateFintuneData(country):
    ans_item = dict()
    with codecs.open(f'data/{country}/Iraq.csv', encoding='utf-8-sig') as f:
    # with codecs.open(f'data/{country}/{country}.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            if row['B_COUNTRY'] == 'Avg':
            # if row['B_COUNTRY'] == 'Avg_First':
                ans_item = {'B_COUNTRY': row['B_COUNTRY'], 'B_COUNTRY_ALPHA': row['B_COUNTRY_ALPHA']}
                for q in q_list:
                    k = 'Q' + q
                    ans_item[k] = int(float(row[k]))
                print('Ans: ', ans_item)
    f.close()

    dir_path = f"data/{country}/Finetune"
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

    with jsonlines.open(f"{dir_path}/WVQ_{country}_test3.jsonl", "a") as writer:
        with open("data/WVQ.jsonl", "r+", encoding="utf8") as f:
            t = 0
            for item in jsonlines.Reader(f):
                prompt = getPrompt(item, t)
                print(item['q_id'], ' ', item['q_content'])
                print(prompt)
                # prompt = translate(prompt)
                # print(prompt)
                ans = ans_item['Q'+item['q_id']]
                if ans < 0:
                    ans = 0 - ans
                new_item = {"messages": [{"role": "system", "content": f"You are an {country} chatbot that know {country} very well."}, 
                                        {"role": "user", "content": prompt}, 
                                        {"role": "assistant", "content": str(ans)}]}
                writer.write(new_item)
                t += 1
        with open("data/data_500_types_mini.jsonl", "r+", encoding="utf8") as f:
            t = 0
            for item in jsonlines.Reader(f):
                prompt = getPrompt(item, t)
                print(item['q_id'], ' ', item['q_content'])
                print(prompt)
                # prompt = translate(prompt)
                # print(prompt)
                ans = ans_item['Q'+item['q_id']]
                if ans < 0:
                    ans = 0 - ans
                new_item = {"messages": [{"role": "system", "content": f"You are an {country} chatbot that know {country} very well."}, 
                                        {"role": "user", "content": prompt}, 
                                        {"role": "assistant", "content": str(ans)}]}
                writer.write(new_item)
                t += 1
    print('ok!')

def getSpecificVerison():
    with jsonlines.open("data/WVQ_Arabic.jsonl", "a") as writer:
        with open("data/WVQ.jsonl", "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                new_content = translate(item['q_content'])
                item['ori_content'] = item['q_content']
                item['q_content'] = new_content
                writer.write(item)

def finetune():
    from openai import OpenAI

    client = OpenAI(api_key="xxx")

    client.files.create(
      file=open("", "rb"),
      purpose="fine-tune"
    )

    client.fine_tuning.jobs.create(
    training_file="", 
    model="gpt-3.5-turbo"
    )

def processLanguage():
    ans_list = []
    with open("data/Finetune/WVQ_arabic_Iraq.jsonl", "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            ans = item["messages"][2]["content"]
            ans_list.append(ans)
    
    n = 0
    with jsonlines.open("data/Finetune/WVQ_arabic_Iraq_L.jsonl", "a") as writer:
        with open("data/Finetune/WVQ_arabic_Iraq_Jordan_L.jsonl", "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                item["messages"][2]["content"] = ans_list[n]
                n += 1
                writer.write(item)
    print('ok!')

def generateData4Llama():
    from datasets import load_dataset
    def formatting_func(example):
        text = f"### Question: {example['messages'][1]['content']}\n ### Answer: {example['messages'][2]['content']}"
        return text
    
    path_name = 'data/Portuguese/Finetune/WVQ_Portuguese'
    dataset = load_dataset('json', data_files=f'{path_name}.jsonl', split='train')

    with jsonlines.open(f'{path_name}_llama.jsonl',mode='a') as writer:
        for item in dataset:
            text = formatting_func(item)
            new_item = {'text': text}
            writer.write(new_item)
    print('ok!')
    
def translateFile(language):
    file_path = f'data/{language}/Finetune/WVQ_{language}'
    lan_dict = {'Arabic': 'Arabic', 'Bengali': 'Bengali', 'China': 'Chinese', 'English': 'English', 'Germany': 'German', 'Korean': 'Korean', 'Portuguese': 'Portuguese', 'Spanish': 'Spanish', 'Turkey': 'Turkish'}
    with jsonlines.open(f"{file_path}_L.jsonl", "a") as writer:
        with open(f"{file_path}.jsonl", "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                content = item["messages"][1]["content"]
                language_t = lan_dict[language]
                trans_content = translate(content, language_t)
                print('Trans: ', trans_content)
                item["messages"][1]["content"] = trans_content
                # new_msg = {"role": "user", "content": trans_content}
                # new_item = {"messages": [item["messages"][0], new_msg, item["messages"][2]]}
                writer.write(item)
    print('ok!')

def mergeFile():
    lan_list = ['Arabic', 'Bengali', 'China', 'English', 'Germany', 'Korean', 'Portuguese', 'Spanish', 'Turkey']
    model_list = ['arabic_Iraq_Jordan', 'Bengali', 'China', 'English', 'Germany', 'Korean', 'Portuguese', 'Spanish', 'Turkey']
    with jsonlines.open(f"data/Finetune/WVQ_all.jsonl", "a") as writer:
        for i in range(len(lan_list)):
            lan = lan_list[i]
            data = model_list[i]
            file_path = f'data/{lan}/Finetune/WVQ_{data}.jsonl'
            with open(file_path, "r+", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    writer.write(item)

    print('ok!!')

