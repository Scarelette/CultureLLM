import jsonlines, os
import openpyxl

workbook = openpyxl.Workbook()
sheet = workbook.active

language_list = ['arabic', 'bengali', 'china', 'english', 'germany', 'korean', 'portuguese', 'spanish', 'turkish']
model_list = ['Jordan_Iraq', 'bengali', 'china', 'english', 'germany', 'korean', 'portuguese', 'spanish', 'turkish']
dir_path = 'results'
for i in range(len(language_list)):
    lan = language_list[i]
    model = model_list[i]
    print('Language: ', lan)
    lan_dir = os.path.join(dir_path, lan)
    # lan_dict = {'Dataset': ['ChatGPT', 'ChatGPT(+50)', 'Ours(Sole)', 'Ours(Whole)', 'Gemini', 'GPT-4']}
    lan_dict = {'Dataset': ['ChatGPT', 'ChatGPT(+L)', 'Ours']}
    file_names = os.listdir(lan_dir)
    sorted_files = sorted(file_names)
    for task_file in sorted_files:
        task_path = os.path.join(lan_dir, task_file)
        if os.path.isfile(task_path):
            task_name = task_file[:-10]
            print(task_name)
            my_dict = dict()
            try:
                with open(task_path, "r+", encoding="utf8") as f:
                    for item in jsonlines.Reader(f):
                        # print('Item: ', item)
                        if item['Model'] not in my_dict.keys():
                            my_dict[item['Model']] = item['f1_score']
                        elif 'Context' in item.keys() and item['Context'] == True and 'PE' not in my_dict.keys():
                            my_dict['PE'] = item['f1_score']
                        else:
                            if item['Model'] == model or item['Model'] == f"{model}_1000" or item['Model'] == f"{model}_L":
                                if item['f1_score'] > my_dict[model]:
                                    my_dict[model] = item['f1_score']
                            elif 'Context' in item.keys() and item['Model'] == 'chatgpt' and item['Context'] == True:
                                if item['f1_score'] < my_dict['PE']:
                                    my_dict['PE'] = item['f1_score']
                            elif item['Model'] == 'cultureLLM':
                                if item['f1_score'] > my_dict[model]:
                                    my_dict[model] = item['f1_score']
                            else:
                                if item['f1_score'] < my_dict[item['Model']]:
                                    my_dict[item['Model']] = item['f1_score']
                # print(my_dict)
                # print('chatgpt: ', my_dict['chatgpt'])
                # print('Ours: ', my_dict[model])
                # print('gemini: ', my_dict['gemini'])
                # print('gpt-4: ', my_dict['gpt-4'])
                # print('\n')
                print(my_dict['PE'])
                print(my_dict[model])
                # print(my_dict['cultureLLM'])
                # print(my_dict['gemini'])
                # print(my_dict['gpt-4'])
                print('')
                # lan_dict[task_name] = [my_dict['chatgpt'], my_dict[f"{model}_50"], my_dict[model], my_dict['cultureLLM'], my_dict['gemini'], my_dict['gpt-4']]
                if f"{model}_L" in my_dict.keys():
                    lan_dict[task_name] = [my_dict['chatgpt'], my_dict[f"{model}_L"], my_dict[model]]
                else:
                    lan_dict[task_name] = [my_dict['chatgpt'], my_dict[model], my_dict[model]]
                print(lan_dict)
            except:
                continue
    header = list(lan_dict.keys())
    sheet.append(header)
    print(lan_dict)
    for i in range(len(lan_dict[task_name])):
        row_data = [lan_dict[key][i] for key in header]
        sheet.append(row_data)
    
    workbook.save('example_dict.xlsx')

print("ok!")



