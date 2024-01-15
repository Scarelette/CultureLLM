# CultureLLM: Fine-tuning Culture-aware Large Language Models with Semantic Data Augmentation

Large language models (LLMs) are reported to be partial to certain cultures and thus suffer from culture difference. While cultural data are often expensive to collect,  existing efforts handles this challenge by prompt engineering or culture-specific pre-training. However, they might overlook the culture specificity and require extensive computing resources. In this paper, we propose CultureLLM, a cost-effective solution to fine-tune culture-aware LLMs. CultureLLM adopts World Value Survey as the seed data and then generates diverse and semantically equivalent training data using the proposed semantic data augmentation approach. Using only $50$ seed samples from WVS with augmented data, we fine-tuned culture-specific LLMs and one unified model (CultureLLM-One) for 9 cultures covering rich and low-resource languages. Extensive experiments on 59 culture-related datasets demonstrate that CultureLLM significantly outperforms various counterparts such as ChatGPT and Gemini Pro with comparable or even better performance than GPT-4. Our human study shows that the generated samples are diverse and semantically equivalent to the original samples, providing an effective solution for LLMs augmentation.

## Install

```bash
pip install jsonlines fire scikit-learn torch==2.0.0 transformers bitsandbytes accelerate
pip install openai
pip install -q -U google-generativeai
```
## Data Augmentation

```bash
python main.py --n 5 --m 10
```
Parameter: 
- n: number of new generated semantic equvalient sentences perserved in the first step in our alogorithm for one seed sample
- m: number of final generated samples perserved in our alogorithm for one seed sample

## Dataset for Fine-tuning and Experiments

/data contains all datasets for fine-tuning and experiments.
/data/WVQ.jsonl contains seed data from World Values Survey.
/data/new_WVQ_500.jsonl contains 500 new generated samples via our data augmentation approach. Those data samples are also be used in our main experiments.
/data/new_WVQ_100.jsonl contains 100 new generated samples via our data augmentation approach.
/data/new_WVQ_1000.jsonl contains 1000 new generated samples via our data augmentation approach.
/data/new_WVQ_sentence_only.jsonl contains 500 new generated samples via the first steps in our data augmentation approach.

Besides, there are nine directories in /data, containing datasets both for fine-tuning and experiments for specific culture. /Finetune contains different versions of training datasets, which are distinguished by the last characters in file names. "50", "150", "no suffix", and "1000" represents fine-tuning via 50, 150, 500 and 1000 new generated samples, respectively. "L" respresents fine-tuning via 500 new generated samples in specific language. "llama" represents fine-tuning file for Llama. "sentence_only" represents fine-tuning via 500 new generated samples by the first steps in our data augmentation approach.

## Fine-tuning your own CultureLLM


## Experiments

Outline the experiments conducted during the project. Include details about the models trained, hyperparameters, and any specific configurations. This section can also include the results of the experiments.

## CultureLLM-Llama-70b-chat

Provide specific details about the CultureLLM Llama 70b chat model. Explain its purpose in your project, how it is utilized, and any important considerations or modifications made.

## Contributing

If you want others to contribute to your project, provide guidelines on how they can do so. Include information on the development environment, coding standards, and the process for submitting pull requests.

## License

Specify the license under which your project is distributed. This is crucial for others who want to use or contribute to your codebase.

## Acknowledgments

Give credit to any external libraries, frameworks, or individuals whose work contributed to your project.

## Contact

Provide contact information or links to relevant channels where users can reach out for support, questions, or collaboration.
```

Remember to replace the placeholder information with the actual details for your project. The README serves as a guide for users and contributors, so make sure it's clear and informative.