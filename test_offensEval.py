import jsonlines, os, re
from sklearn.metrics import f1_score, accuracy_score
import fire, time, random
import pathlib
import textwrap
from llm_response import getResponse

context_dict = dict()
with open("data/culture_context.jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        for key in item.keys():
            context_dict[key] = item[key]

def getModel(language):
    if language.lower() == 'arabic':
        model_dict = {'Jordan_Iraq': 'xxx',  # best
                    'chatgpt': 'gpt-3.5-turbo',
                    'Jordan_Iraq_L': 'xxx',
                    'llama_Arabic': 'xxx',
                    'llama': 'xxx',
                    'Jordan_Iraq_50': 'xxx',
                    'Jordan_Iraq_100': 'xxx',
                    'Jordan_Iraq_1000': 'xxx',
                    'Jordan_Iraq_s': 'xxx'}
    elif language.lower() == 'turkish':
        model_dict = {'turkish': 'xxx',
                      'turkish_L': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo',
                    'llama_Turkey': 'xxx',
                    'llama': 'xxx',
                    'turkish_50': 'xxx',
                    'turkish_100': 'xxx',
                    'turkish_1000': 'xxx',
                    'turkish_s': 'xxx'}
    elif language.lower() == 'greek':
        model_dict = {'greek': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo'}
    elif language.lower() == 'germany':
        model_dict = {'germany': 'xxx',
                      'germany_L': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo',
                    'llama_Germany': 'xxx',
                    'llama': 'xxx',
                    'germany_50': 'xxx',
                    'germany_100': 'xxx',
                    'germany_1000': 'xxx',
                    'germany_s': 'xxx'}
    elif language.lower() == 'spanish':
        model_dict = {'spanish': 'xxx',
                      'spanish_L': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo',
                    'llama_Spanish': 'xxx',
                    'llama': 'xxx',
                    'spanish_50': 'xxx',
                    'spanish_100': 'xxx',
                    'spanish_1000': 'xxx',
                    'spanish_s': 'xxx'}
    elif language.lower() == 'bengali':
        model_dict = {'bengali': 'xxx',
                      'bengali_L': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo',
                    'llama_Bengali': 'xxx',
                    'llama': 'xxx',
                    'bengali_50': 'xxx',
                    'bengali_100': 'xxx',
                    'bengali_1000': 'xxx',
                    'bengali_s': 'xxx'}
    elif language.lower() == 'chinese':
        model_dict = {'china': 'xxx',
                      'china_L': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo',
                    'llama_China': 'xxx',
                    'llama': 'xxx',
                    'china_50': 'xxx',
                    'china_100': 'xxx',
                    'china_1000': 'xxx',
                    'china_s': 'xxx'}
    elif language.lower() == 'english':
        model_dict = {'english': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo',
                    'llama_English': 'xxx',
                    'llama': 'xxx',
                    'english_50': 'xxx',
                    'english_100': 'xxx',
                    'english_1000': 'xxx',
                    'english_s': 'xxx'}
    elif language.lower() == 'korean':
        model_dict = {'korean': 'xxx',
                      'korean_L': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo',
                    'llama_Korean': 'xxx',
                    'llama': 'xxx',
                    'korean_50': 'xxx',
                    'korean_100': 'xxx',
                    'korean_1000': 'xxx',
                    'korean_s': 'xxx'}
    elif language.lower() == 'portuguese':
        model_dict = {'portuguese': 'xxx',
                      'portuguese_L': 'xxx',
                      'chatgpt': 'gpt-3.5-turbo',
                    'llama_Portuguese': 'xxx',
                    'llama': 'xxx',
                    'portuguese_50': 'xxx',
                    'portuguese_100': 'xxx',
                    'portuguese_1000': 'xxx',
                    'portuguese_s': 'xxx'}

    return model_dict

def getPrompt(input, task, input_1=None):
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
    # if task == 'offensive_detect' or task == 'offensive_detect_osact4' or task == 'offensive_detect_mp' or task == 'offensive_detect_osact5':
        prompt = f"if the following sentence is offensive, just say \"OFF\", otherwise, say \"NOT_OFF\" without explanation: {input}"
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        prompt = f"if the following sentence has hate speech, just say \"HS\", otherwise, say \"NOT_HS\" without explanation: {input}"
    elif task == 'vulgar_detect_mp':
        prompt = f"if the following sentence has vulgar speech, just say \"Vulgar\", otherwise, say \"NOT_Vulgar\" without explanation: {input}"
    elif task == 'spam_detect':
        prompt = f"if the following sentence is spam tweet, just say \"Spam\", otherwise, say \"NOT_Spam\" without explanation: {input}"
    elif task == 'hate_detect_fine-grained':
        prompt = f"if the following sentence doesn't have hate speech, just say \"NOT_HS\", otherwise, label the hate speech with \"HS1\"(Race), \"HS2\"(Religion), \"HS3\"(Ideology), \"HS4\"(Disability), \"HS5\"(Social Class), \"HS6\"(Gender) without explanation: {input}"
    elif task == 'offensive_detect_finegrained':
        prompt = f"if the following sentence doesn't have offensive speech, just say \"non\", otherwise, label the offensive speech with \"prof\"(profanity, or non-targeted offense), \"grp\"(offense towards a group), \"indv\"(offense towards an individual), \"oth\"(ffense towards an other (non-human) entity, often an event or organization) without explanation: {input}"
    elif task == 'hate_off_detect':
        prompt = f"if the following sentence has hate speech or offensive content, just say \"HOF\", otherwise, say \"NOT\" without explanation: {input}"
    elif task == 'stereotype_detect' or task == 'mockery_detect' or task == 'insult_detect' or task == 'improper_detect' or task == 'aggressiveness_detect' or task == 'toxicity_detect' or task == 'negative_stance_detect' or task == 'homophobia_detect' or task == 'racism_detect' or task == 'misogyny_detect' or task == 'threat_detect':
        entity = task[:-7]
        prompt = f"if the following sentence has {entity} speech, just say \"1\", otherwise, say \"0\" without explanation: {input}"
    elif task == 'bias_on_gender_detect' or task == 'hostility_directness_detect':
        entity = task[:-7]
        entity = entity.replace('_', ' ')
        prompt = f"if the following speech expressing {entity}, just say \"1\", otherwise, say \"0\" without explanation: {input}"
    elif task == 'hate_offens_detect':
        prompt = f"if the following sentence contains hate speech, just say \"0\", else if contains offensive language, say \"1\", otherwise, say \"2\" without explanation: {input}"
    else:
        prompt = input

    return prompt

def getDataPath(language, task):
    pre = '/mnt/mydata/cultureLLM/'
    if task == 'GSM8K':
        path = 'data/forget/GSM8K_test.jsonl'
    if language.lower() == 'arabic':
        if task == 'offensive_detect':
            path = "data/Arabic/OffensEval2020/OffensEval.jsonl"
        elif task == 'hate_detect_osact4':
            path = "data/Arabic/OSACT4/dev_data.jsonl"
        elif task == 'offensive_detect_osact4':
            path = "data/Arabic/OSACT4/dev_data_offens.jsonl"
        elif task == 'offensive_detect_mp':
            path = "data/Arabic/MP/offens.jsonl"
        elif task == 'hate_detect_mp':
            path = "data/Arabic/MP/hateSpeech.jsonl"
        elif task == 'vulgar_detect_mp':
            path = "data/Arabic/MP/VulgarSpeech.jsonl"
        elif task == 'spam_detect':
            path = "data/Arabic/SpamDetect/span_detect_2.jsonl"
        elif task == 'offensive_detect_osact5':
            path = "data/Arabic/OSACT5/offens.jsonl"
        elif task == 'hate_detect_osact5':
            path = "data/Arabic/OSACT5/hateSpeech.jsonl"
        elif task == 'hate_detect_fine-grained':
            path = "data/Arabic/OSACT5/hate_Finegrained.jsonl"
    elif language.lower() == 'turkish':
        if task == 'offensive_detect':
            path = "data/Turkey/OffensEval2020/OffensEval.jsonl"
        elif task == 'offensive_detect_corpus':
            path = "data/Turkey/offenseCorpus/offens.jsonl"
        elif task == 'offensive_detect_finegrained':
            path = "data/Turkey/offenseCorpus/offens_fine-graind.jsonl"
        elif task == 'offensive_detect_kaggle':
            path = "data/Turkey/offenssDetect-kaggle/turkish_tweets_2020.jsonl"
        elif task == 'offensive_detect_kaggle2':
            path = "data/Turkey/offensDetect-kaggle2/test.jsonl"
        elif task == 'abusive_detect':
            path = "data/Turkey/ATC/fold_0_test.jsonl"
        elif task == 'spam_detect':
            path = "data/Turkey/TurkishSpam/trspam.jsonl"
    elif language.lower() == 'greek':
        if task == 'offensive_detect':
            path = "data/Greece/OffensEval2020/OffensEval.jsonl"
        elif task == 'offensive_detect_g':
            path = 'data/Greece/gazzetta/G-TEST-S-preprocessed.jsonl'
    elif language.lower() == 'germany':
        if task == 'hate_detect':
            path = 'data/Germany/IWG_hatespeech_public/german_hatespeech_refugees_2.jsonl'
        elif task == 'hate_off_detect':
            path = 'data/Germany/HASOC/hate_off_detect.jsonl'
        elif task == 'offensive_detect_eval':
            path = 'data/Germany/GermEval/germeval2018.jsonl'
        elif task == 'hate_detect_check':
            path = 'data/Germany/MHC/hatecheck_cases_final_german.jsonl'
        elif task == 'hate_detect_iwg_1':
            path = 'data/Germany/IWG_hatespeech_public/german_hatespeech_refugees_1.jsonl'
    elif language.lower() == 'spanish':
        if task == 'offensive_detect_ami':
            path = 'data/Spanish/AMI IberEval 2018_offens/data-2.jsonl'
        elif task == 'offensive_detect_mex_a3t':
            path = 'data/Spanish/MEX-A3T_offens/data-2.jsonl'
        elif task == 'offensive_detect_mex_offend':
            path = 'data/Spanish/OffendES_offens/data-2.jsonl'
        elif task == 'hate_detect_eval':
            path = 'data/Spanish/HateEval 2019_HS/data-2.jsonl'
        elif task == 'hate_detect_haterNet':
            path = 'data/Spanish/HaterNet_HS/data-2.jsonl'
        elif task == 'stereotype_detect':
            path = 'data/Spanish/DETOXIS 2021/stereotype.jsonl'
        elif task == 'mockery_detect':
            path = 'data/Spanish/DETOXIS 2021/mockery.jsonl'
        elif task == 'insult_detect':
            path = 'data/Spanish/DETOXIS 2021/insult.jsonl'
        elif task == 'improper_detect':
            path = 'data/Spanish/DETOXIS 2021/improper_language.jsonl'
        elif task == 'aggressiveness_detect':
            path = 'data/Spanish/DETOXIS 2021/aggressiveness.jsonl'
        elif task == 'negative_stance_detect':
            path = 'data/Spanish/DETOXIS 2021/negative_stance.jsonl'
    elif language.lower() == 'bengali':
        if task == 'offensive_detect_1':
            path = 'data/Bengali/Trac2-Task1-Aggresion/aggression-data-2.jsonl'
        elif task == 'offensive_detect_2':
            path = 'data/Bengali/Trac2-Task2-Misogynistic/Misogynistic-data-2.jsonl'
        elif task == 'offensive_detect_3':
            path = 'data/Bengali/BAD-Bangla-Aggressive-Text-Dataset/data-2.jsonl'
        elif task == 'hate_detect_religion':
            path = 'data/Bengali/Bengali hate speech dataset/religion_data-2.jsonl'
        elif task == 'threat_detect':
            path = 'data/Bengali/Bangla-Abusive-Comment-Dataset/threat.jsonl'
        elif task == 'racism_detect':
            path = 'data/Bengali/Bangla-Abusive-Comment-Dataset/racism.jsonl'   
    elif language.lower() == 'chinese':
        if task == 'spam_detect':
            path = 'data/China/Chinese-Camouflage-Spam-dataset/data-2.jsonl'
        elif task == 'bias_on_gender_detect':
            path = 'data/China/CDial-Bias/gender-2.jsonl'
    elif language.lower() == 'korean':
        if task == 'hate_detect_3':
            path = 'data/Korean/K-MHaS/data-2.jsonl'
        elif task == 'hate_detect_6':
            path = 'data/Korean/Korean-Hate-Speech-Detection/data-2.jsonl'
        elif task == 'hate_detect_7':
            path = 'data/Korean/KoreanHateSpeechdataset/data-2.jsonl'
        elif task == 'abusive_detect':
            path = 'data/Korean/AbuseEval/data-2.jsonl'
        elif task == 'abusive_detect_2':
            path = 'data/Korean/CADD/data-2.jsonl'
        elif task == 'abusive_detect_4':
            path = 'data/Korean/Waseem/data-2.jsonl'
    elif language.lower() == 'portuguese':
        if task == 'offensive_detect_2':
            path = "data/Portuguese/OffComBR/data.jsonl"
        elif task == 'offensive_detect_3':
            path = "data/Portuguese/HateBR/data-2.jsonl"
        elif task == 'homophobia_detect':
            path = 'data/Portuguese/ToLD-Br/homophobia.jsonl'
        elif task == 'misogyny_detect':
            path = 'data/Portuguese/ToLD-Br/misogyny.jsonl'
        elif task == 'insult_detect':
            path = 'data/Portuguese/ToLD-Br/insult.jsonl'
    elif language.lower() == 'english':
        if task == 'hate_detect_2':
            path = 'data/English/MLMA hate speech/data-2.jsonl'
        elif task == 'hostility_directness_detect':
            path = 'data/English/MLMA hate speech/directness.jsonl'
        elif task == 'hate_offens_detect':
            path = 'data/English/hate-speech-and-offensive-language/data.jsonl'
        elif task == 'offensive_detect_easy':
            path = 'data/English/SOLID/test_a_tweets_easy.jsonl'
        elif task == 'toxicity_detect':
            path = 'data/English/Toxic Comment Classification Challenge/toxic.jsonl'
        elif task == 'threat_detect':
            path = 'data/English/Toxic Comment Classification Challenge/threat.jsonl'
    return path

def postprocess(task, output):
    
    if output == '':
        return output
    
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        if 'not' in output.lower():
            label = 'NOT'
        elif 'off' in output.lower():
            label = 'OFF'
        else:
            label = output
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        if 'not' in output.lower():
            label = 'NOT_HS'
        elif 'hs' in output.lower():
            label = 'HS'
        else:
            label = output
    elif task == 'vulgar_detect_mp':
        if 'not' in output.lower():
            label = '-'  
        elif 'vulgar' in output.lower():
            label = 'V'
        else:
            label = output
    elif task == 'spam_detect':
        if 'no' in output.lower():
            label = 'Ham'
        elif 'spam' in output.lower():
            label = 'Spam'
        else:
            label = output
    elif task == 'hate_detect_fine-grained':
        if 'no' in output.lower():
            label = 'NOT_HS'
        elif '1' in output.lower():
            label = 'HS1'
        elif '2' in output.lower():
            label = 'HS1'
        elif '3' in output.lower():
            label = 'HS1'
        elif '4' in output.lower():
            label = 'HS1'
        elif '5' in output.lower():
            label = 'HS1'
        elif '6' in output.lower():
            label = 'HS1'
        else:
            label = output
    elif task == 'offensive_detect_finegrained':
        if 'no' in output.lower():
            label = 'non'
        elif 'prof' in output.lower():
            label = 'prof'
        elif 'grp' in output.lower():
            label = 'grp'
        elif 'indv' in output.lower():
            label = 'indv'
        elif 'oth' in output.lower():
            label = 'oth'
        else:
            label = output
    elif task == 'hate_off_detect':
        if 'no' in output.lower():
            label = 'NOT'
        elif 'hof' in output.lower():
            label = 'HOF'
        else:
            label = output
    elif task == 'stereotype_detect' or task == 'mockery_detect' or task == 'insult_detect' or task == 'improper_detect' or task == 'aggressiveness_detect' or task == 'toxicity_detect' or task == 'negative_stance_detect' or task == 'bias_on_gender_detect' or task == 'homophobia_detect' or task == 'racism_detect' or task == 'misogyny_detect' or task == 'threat_detect' or task == 'hostility_directness_detect':
        if '1' in output.lower():
            label = '1'
        elif '0' in output.lower():
            label = '0'
        else:
            label = output
    elif task == 'hate_offens_detect':
        if '0' in output.lower():
            label = '0'
        elif '1' in output.lower():
            label = '1'
        else:
            label = '2'
    elif task == 'GSM8K':
        pred = output.lower()
        pattern = r'\d+,\d+'
        matches = re.findall(pattern, pred)
        formatted_numbers = []
        has = 0
        for match in matches:
            if ',' in match:
                formatted_numbers.append(match.replace(',', ''))
                has = 1
            else:
                formatted_numbers.append(match)
        if has == 0:
            pattern = r'\d+'
            matches = re.findall(pattern, pred)
        else:
            matches = formatted_numbers
        
        if len(matches) > 0:
            label = matches[-1]
        else:
            label = output

    return label

def computeMetrics(task, gt_list, pred_list):
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        final_f1_score = f1_score(gt_list, pred_list, labels=['OFF', 'NOT'], average='macro')
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['HS', 'NOT_HS'], average='macro')
    elif task == 'vulgar_detect_mp':
        final_f1_score = f1_score(gt_list, pred_list, labels=['V', '-'], average='macro')
    elif task == 'spam_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['Spam', 'Ham'], average='macro')
    elif task == 'hate_detect_fine-grained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['NOT_HS', 'HS1', 'HS2', 'HS3', 'HS4', 'HS5', 'HS6'], average='macro')
    elif task == 'offensive_detect_finegrained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['non', 'prof', 'grp', 'indv', 'oth'], average='macro')
    elif task == 'hate_off_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['HOF', 'NOT'], average='macro')
    elif task == 'stereotype_detect' or task == 'mockery_detect' or task == 'insult_detect' or task == 'improper_detect' or task == 'aggressiveness_detect' or task == 'toxicity_detect' or task == 'negative_stance_detect' or task == 'bias_on_gender_detect' or task == 'homophobia_detect' or task == 'racism_detect' or task == 'misogyny_detect' or task == 'threat_detect' or task == 'hostility_directness_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1'], average='macro')
    elif task == 'hate_offens_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1', '2'], average='macro')
    elif task == 'GSM8K':
        final_f1_score = accuracy_score(gt_list, pred_list)

    return final_f1_score

def run(language, model_text, task, context, close=False):
    gt_list = []

    path = getDataPath(language, task)
    
    # num = 0
    with open(path, "r+", encoding="utf8") as f:
        # if num < 1000:
        for item in jsonlines.Reader(f):
            if task == 'GSM8K':
                label = item['answer']
            else:
                label = item['label']
            gt_list.append(label)
        # num += 1

    pred_list = []

    if model_text == 'cultureLLM':
        model = 'xxx'
    else:
        model_dict = getModel(language)

        if model_text in model_dict.keys():
            model = model_dict[model_text]
        else:
            model = model_text
    
    llama_tokenizer = None
    llama_model = None
    if 'llama'in model.lower() and 'api' not in model.lower():
        from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig, AutoModelWithLMHead, AutoTokenizer, BitsAndBytesConfig
        import torch
        # llama_tokenizer = LlamaTokenizer.from_pretrained(model)
        # llama_model = LlamaForCausalLM.from_pretrained(model, device_map="auto")
        compute_dtype = getattr(torch, "float16")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        
        # Path for llama model
        base_model = 'xxx'
        max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
        llama_model = LlamaForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            # device_map={"": 0}
            device_map="auto",
            max_memory=max_memory
        )
        llama_model.quantization_config = quant_config
        llama_model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        llama_tokenizer = tokenizer

    with open(path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            if task == 'offensive_detect_mp' or task == 'hate_detect_mp' or task == 'vulgar_detect_mp':
                if 'comment' in item.keys():
                    input = item['comment']
                else:
                    input = item['data']
            elif 'tweet' in item.keys():
                input = item['tweet']
            elif task == 'GSM8K':
                input = item['question']
            else:
                input = item['data']
            print('Input: ', input)
            
            prompt = getPrompt(input, task)
            print('P: ', prompt)
            if close == True:
                output = getResponse(prompt, model, 'no', context, llama_model, llama_tokenizer)
            else:
                output = getResponse(prompt, model, language, context, llama_model, llama_tokenizer)

            if output == None:
                output = ''
            print('Output: ', output)

            label = postprocess(task, output)

            pred_list.append(label)

    final_f1_score = computeMetrics(task, gt_list, pred_list)
    
    print('F1 score: ', final_f1_score)

    language = language.lower()
    dir_path = f'results/{language}'
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

    with jsonlines.open(f'{dir_path}/{task}_res.jsonl',mode='a') as writer:
        item = {'Model': model_text, "f1_score": final_f1_score, 'Context': context}
        writer.write(item)

if __name__ == '__main__':
    fire.Fire(run)    




