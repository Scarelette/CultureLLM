from string import punctuation
import jsonlines
import random, fire

def perplexity(sentence, uni_gram_dict, bi_gram_dict):
    sentence_cut = sentence.split()
    V = len(uni_gram_dict)
    sentence_len = len(sentence_cut)
    p=1     # 概率初始值
    k=0.5   # ngram 的平滑值，平滑方法：Add-k Smoothing （k<1）
    for i in range(sentence_len-1):
        two_word = "".join(sentence_cut[i:i+2])
        p *=(bi_gram_dict.get(two_word,0)+k)/(uni_gram_dict.get(sentence_cut[i],0)+k*V)
    
    # print(p)
    return pow(1/p, 1/sentence_len)

def compute_ppl(s):
    dicts={i:'' for i in punctuation}
    punc_table=str.maketrans(dicts)
    s=s.translate(punc_table)
    uni_dict = dict()
    bi_dict = dict()
    words = s.split()
    for w in words:
        if w not in uni_dict.keys():
            uni_dict[w] = 1
        else:
            uni_dict[w] += 1
    for i in range(len(words)-1):
        bi_gram = words[i] + ' ' + words[i+1]
        if bi_gram not in bi_dict.keys():
            bi_dict[bi_gram] = 1
        else:
            bi_dict[bi_gram] += 1

    ppl = perplexity(s, uni_dict, bi_dict)

    return ppl

def run(data_file):
        
    mean_ppl = 0.0
    num = 0
    with open(data_file, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            num += 1
            s = item['q_content']         
            s_ppl = compute_ppl(s)
            mean_ppl += s_ppl

    mean_ppl /= num
    print('Mean ppl: ', mean_ppl)

if __name__ == '__main__':
    fire.Fire(run)  