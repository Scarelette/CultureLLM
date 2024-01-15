import openai, fire
import math, jsonlines, time

def getEmbedding(s):
    output = None
    times = 0
    while output is None and times <= 10:
        try:
            times += 1  
            response = openai.Embedding.create(
                api_key="xxx",
                input=s,
                model="text-embedding-ada-002"
                )
            output = response['data'][0]['embedding']
        except Exception as e:
            print(e)
            print('Retrying...')
            time.sleep(5)
    if times >= 10:
        print('Failed! Model Input: ', s)
        output = ''
            
    return output

def computeDis(s1, s2):
    embed_1 = getEmbedding(s1)
    embed_2 = getEmbedding(s2)

    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(embed_1, embed_2)]))


def run(new_data_file, origin_data_file):
    num = 0
    diverse_gain = 0
    file_path = new_data_file
    with open(file_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            print(num)
            num += 1
            content = item['q_content']
            dis_list = []
            with open(origin_data_file, "r+", encoding="utf8") as f2:
                for ori_item in jsonlines.Reader(f2):
                    ori_content = ori_item['q_content']
                    cur_dis = computeDis(content, ori_content)
                    dis_list.append(cur_dis)
                    print('Cur dis: ', cur_dis)
            final_dis = min(dis_list)
            print('Final dis: ', final_dis)
            diverse_gain += final_dis
    diverse_gain /= num

    print('Diverse Gain: ', diverse_gain)

    with jsonlines.open('results/diverse_gain.jsonl',mode='a') as writer:
        item = {'file': file_path, 'diverse_gain': diverse_gain}
        writer.write(item)

if __name__ == '__main__':
    fire.Fire(run)  


# computeDiverse()



