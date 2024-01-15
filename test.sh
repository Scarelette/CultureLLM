# arabic
for task in 'offensive_detect' 'offensive_detect_osact4' 'offensive_detect_mp' 'offensive_detect_osact5' 'hate_detect_osact4' 'hate_detect_mp' 'hate_detect_osact5' 'vulgar_detect_mp'  'spam_detect'  'hate_detect_fine-grained'
do 
    python test_offensEval.py --language arabic --model_text Jordan_Iraq --task $task --context False
    python test_offensEval.py --language arabic --model_text chatgpt --task $task --context False
    python test_offensEval.py --language arabic --model_text gpt-4 --task $task --context False
done


# bengali
for task in 'hate_detect_religion' 'offensive_detect_1' 'offensive_detect_2' 'offensive_detect_3' 'racism_detect' 'threat_detect' 
do 
    python test_offensEval.py --language bengali --model_text bengali --task $task --context False
    python test_offensEval.py --language bengali --model_text chatgpt --task $task --context False
    python test_offensEval.py --language bengali --model_text gpt-4 --task $task --context False
done


# china
for task in 'bias_on_gender_detect' 'spam_detect'
do 
    python test_offensEval.py --language china --model_text china --task $task --context False
    python test_offensEval.py --language china --model_text chatgpt --task $task --context False
    python test_offensEval.py --language china --model_text gpt-4 --task $task --context False
done



for task in 'bias_on_gender_detect' 'spam_detect'
do 
    python test_offensEval.py --language china --model_text china --task $task --context False
    python test_offensEval.py --language china --model_text chatgpt --task $task --context False
    python test_offensEval.py --language china --model_text gpt-4 --task $task --context False
done


# english
for task in 'hate_detect_2' 'hate_offens_detect' 'hostility_directness_detect' 'offensive_detect_easy' 'threat_detect' 'toxicity_detect'
do 
    python test_offensEval.py --language english --model_text english --task $task --context False
    python test_offensEval.py --language english --model_text chatgpt --task $task --context False
    python test_offensEval.py --language english --model_text gpt-4 --task $task --context False
done


# germany
for task in 'hate_detect' 'hate_off_detect' 'hate_detect_iwg_1' 'hate_detect_check' 'offensive_detect_eval'
do 
    python test_offensEval.py --language germany --model_text germany --task $task --context False
    python test_offensEval.py --language germany --model_text chatgpt --task $task --context False
    python test_offensEval.py --language germany --model_text gpt-4 --task $task --context False
done

# korean
for task in 'abusive_detect' 'abusive_detect_2' 'abusive_detect_4' 'hate_detect_3' 'hate_detect_6' 'hate_detect_7'
do 
    python test_offensEval.py --language korean --model_text korean --task $task --context False
    python test_offensEval.py --language korean --model_text chatgpt --task $task --context False
    python test_offensEval.py --language korean --model_text gpt-4 --task $task --context False
done

# portuguese
for task in 'homophobia_detect' 'insult_detect' 'misogyny_detect' 'offensive_detect_2' 'offensive_detect_3'
do 
    python test_offensEval.py --language portuguese --model_text portuguese --task $task --context False
    python test_offensEval.py --language portuguese --model_text chatgpt --task $task --context False
    python test_offensEval.py --language portuguese --model_text gpt-4 --task $task --context False
done

# spanish
for task in 'offensive_detect_ami' 'offensive_detect_mex_a3t' 'offensive_detect_mex_offend' 'hate_detect_eval' 'hate_detect_haterNet' 'stereotype_detect' 'mockery_detect' 'insult_detect' 'improper_detect' 'aggressiveness_detect'  'negative_stance_detect'
do 
    python test_offensEval.py --language spanish --model_text spanish --task $task --context False
    python test_offensEval.py --language spanish --model_text chatgpt --task $task --context False
    python test_offensEval.py --language spanish --model_text gpt-4 --task $task --context False
done

# turkish
for task in 'offensive_detect' 'offensive_detect_corpus' 'offensive_detect_finegrained' 'offensive_detect_kaggle' 'offensive_detect_kaggle2' 'abusive_detect' 'spam_detect'
do 
    python test_offensEval.py --language turkish --model_text turkish --task $task --context False
    python test_offensEval.py --language turkish --model_text chatgpt --task $task --context False
    python test_offensEval.py --language turkish --model_text gpt-4 --task $task --context False
done

