import jsonlines, os, re
from sklearn.metrics import f1_score, accuracy_score
import fire, time, random
import pathlib
import textwrap

def getResponse(prompt, model_text, country, context, llama_model=None, llama_tokenizer=None):
    def llama_prompt(prompt):
        return f"<s>[INST] <<SYS>>\nYou are an {country} chatbot that know {country} very well.\n<</SYS>>\n\n{prompt} [/INST]"
    if 'llama' in model_text.lower():
        if 'api' in model_text.lower():
            import ssl, json, urllib.request
            def allowSelfSignedHttps(allowed):
                # bypass the server certificate verification on client side
                if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
                    ssl._create_default_https_context = ssl._create_unverified_context

            allowSelfSignedHttps(True)

            data = {
                    "input_data": {
                "input_string": [
                {
                    "role": "user",
                    "content": prompt
                },
                ],
                "parameters": {
                "max_length": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "max_new_tokens": 20
                }
            }}

            body = str.encode(json.dumps(data))
            # print(body)

            url = 'https://gcrllm2-70b-chat.westus3.inference.ml.azure.com/score'
            # Replace this with the primary/secondary key or AMLToken for the endpoint
            api_key = 'xxx'
            if not api_key:
                raise Exception("A key should be provided to invoke the endpoint")

            # The azureml-model-deployment header will force the request to go to a specific deployment.
            # Remove this header to have the request observe the endpoint traffic rules
            headers = {'Content-Type': 'application/json', 'Authorization': (
                'Bearer ' + api_key), 'azureml-model-deployment': 'llama-2-70b-chat-13'}

            req = urllib.request.Request(url, body, headers)

            try:
                response = urllib.request.urlopen(req)
                result = response.read()
                json_string = result.decode('utf-8')
                my_dict = json.loads(json_string)
                output = my_dict['output'].strip()
                time.sleep(1)

            except urllib.error.HTTPError as error:
                print("The request failed with status code: " + str(error.code))
                output = ''
            return output
        else:
            prompt = llama_prompt(prompt)
            input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = llama_model.generate(input_ids, max_new_tokens=200, temperature=0.7, do_sample=True)
            
            out_text = llama_tokenizer.decode(outputs[0])
            out_text = out_text.replace('<pad>', '')
            out_text = out_text.replace('</s>', '')
            out_text = out_text.replace('<s>', '')
            out_text = out_text[len(prompt):]
            out_text = out_text.strip()

            return out_text
    elif 'gemini' in model_text.lower():
        import google.generativeai as genai
        import urllib.request
        import json
        import ssl
        key_pool = ['xxx']
        
        key = random.choice(key_pool)
        genai.configure(api_key=key)
        # genai.configure(api_key='AIzaSyAwihsMaPr-Wiug6fdy-rkmMIQzktjgMMI')
  
        model = genai.GenerativeModel('gemini-pro')
        # print('ok!!!')

        output = None
        times = 0
        while output==None and times <= 2:
            try:
                times += 1  
                response = model.generate_content(
                    prompt, 
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE",
                        }
                    ],
                    generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    max_output_tokens=20,
                    temperature=1.0))
                output = response.text
            except Exception as e:
                output = ''
                print(response.prompt_feedback)
                print('Retrying...')
                time.sleep(60)
        if times >= 2:
            print('Failed! Model Input: ', prompt)
            output = ''
        
        return output
    else:
        from openai import OpenAI

        if country != 'no':
            if context == False:
                msg = [{"role": "system", "content": f"You are an {country} chatbot that know {country} very well."},
                        {"role": "user", "content": prompt}]
            else:
                context = context_dict[country]
                msg = [{"role": "system", "content": f"You are an {country} chatbot that know {country} very well. {context}"},
                        {"role": "user", "content": prompt}]

        else:
            msg = [{"role": "user", "content": prompt}]
        # print('Msg: ', msg)
        client = OpenAI(api_key="xxx")

        # print('MSG: ', msg)

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
