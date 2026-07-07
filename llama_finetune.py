import os, fire
import inspect
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model


## added code - Daniel Benjamin
# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv() 

# accessing and printing value
print(os.getenv("HF_TOKEN"))


# Model from Hugging Face hub or Model path
base_model = ''

def run(base_model, new_model, data_files, resume_from_checkpoint=None):

    dataset = load_dataset('json', data_files=data_files, split='train')

    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        # device_map={"": 0}
        device_map="auto",
        max_memory=max_memory
    )
    model.quantization_config = quant_config
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    output_dir = os.path.abspath(
        os.path.expanduser(os.getenv("FINETUNE_OUTPUT_DIR", "./results-finetune"))
    )
    os.makedirs(output_dir, exist_ok=True)

    training_params = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        #group_by_length=True,
        train_sampling_strategy="group_by_length",
        lr_scheduler_type="constant",
        report_to="tensorboard"
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "peft_config": peft_params,
        #"dataset_text_field": "text",
        #"max_seq_length": None,
        "args": training_params,
    }

    # Pass only the kwargs accepted by the installed TRL version.
    sft_init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "packing" in sft_init_params:
        trainer_kwargs["packing"] = False
    if "processing_class" in sft_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)

def eval():
    prompt = "Who is Leonardo Da Vinci?"
    tokenizer = LlamaTokenizer.from_pretrained("")
    model = LlamaForCausalLM.from_pretrained("", device_map="auto")
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])


if __name__ == '__main__':
    fire.Fire(run)  



