"""
python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style sarcasm_more

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style formality_informal

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style voice_passive

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style missspell

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style grade_elementary

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style function_more

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style length_short

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style length_long

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style sarcasm_less

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style formality_formal

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style voice_active

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json  \
    --style grade_highschool

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json  \
    --style function_less

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json  \
    --style type_persuasive

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json  \
    --style type_expository

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json  \
    --style type_narrative

python3 -m style_training.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token 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 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json  \
    --style type_descriptive
"""

import os
os.environ['TRANSFORMERS_CACHE'] = '../cache/'
os.environ['HF_HOME'] = '../cache/'
os.environ['HUGGINGFACE_HUB_CACHE'] = '../cache/'

from IPython import embed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import math

from datetime import datetime
time = datetime.now()    
date_time = time.strftime("%m-%d-%Y_%H:%M:%S")

from peft import (
    LoraConfig,
    PeftConfig,
    TaskType,
    PeftModel,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments
)
import torch
from datasets import Dataset
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from huggingface_hub import login
import json
import sys
from sklearn.model_selection import train_test_split

def main(args):
    output_dir = os.path.join(args.output_dir,date_time)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"\n\t*\tSaving to {output_dir}\n")
    set_seed(args.seed)

    # Load the data
    data = json.load(open(args.data_path, "r"))
    print(f"List of available styles: {list(data.keys())}")
    if args.style not in data.keys():
        print("Passed in style not found in data. Exiting")
        sys.exit()

    all_origs = []
    all_rewrites = []
    for k in data[args.style]:        
        all_origs.extend([c['content'] for c in data[args.style][k]['originals']])
        all_rewrites.extend(data[args.style][k]['generations'])

    # Split into train and dev set
    train_data, eval_data = train_test_split(
        list(zip(all_origs, all_rewrites)),
        test_size=0.15, 
        random_state=args.seed)
    
    train_original, train_rewrite = zip(*train_data)
    eval_original, eval_rewrite = zip(*eval_data)

    train_dataset = Dataset.from_dict({
        "original": train_original,
        "rewrite": train_rewrite}
        )
    eval_dataset = Dataset.from_dict({
        "original": eval_original,
        "rewrite": eval_rewrite
    })

    # Set the huggingface token
    if args.hf_token:
        login(args.hf_token)
    
    # Load in (toy) data
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    print(f"\n\t*\tModels are on {device}\n")

    # Initialize Lora Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        bias = "none")
    # TODO fix bug where the EOS token is not appearing
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              add_eos_token=True)
    if not tokenizer.pad_token: # Set pad token if it doesn't exist
        tokenizer.pad_token = tokenizer.eos_token 
    # model = AutoModelForCausalLM.from_pretrained(args.model)

    """
    Three ways to load peft config:
    * AutoModelForCausalLM.from_pretrained(mmodel, peft_config = peft_config)
    * PeftModel.from_pretrained(model, peft_config)
    * peft.get_peft_model(model, peft_config)
    """
    # peft_model = get_peft_model(model, peft_config)
    # print(f"\n\t*\t{peft_model.print_trainable_parameters()}\n") # Print out the trainable parameters
    # del peft_model # We reinstantiate it during training

    steps = len(train_data)/(args.train_batch_size*torch.cuda.device_count())
    # Save every quarter epoch
    save_steps = math.ceil(steps / args.save_ratio)
    print(f"\n\t*\tSave steps is {save_steps}\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        report_to="tensorboard",
        evaluation_strategy='steps',
        num_train_epochs=args.epochs,
        eval_steps = save_steps,
        save_steps = save_steps,
        logging_steps = save_steps,
        lr_scheduler_type = 'linear',
        seed = args.seed,
        warmup_ratio = 0.1,
    )

    # def add_eos(batch):
    #     print(list(batch.keys()))
    #     outputs=[]
    #     for b in batch["text"]:
    #         outputs.append(b + " <|end_of_text|>")
    #     return outputs

    # # TODO hotfix remove this add_eos when llama tokenizer fixed
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['original'])):
            text = f"### Original: {example['original'][i]}\n ### Rewrite: {example['rewrite'][i]} <|end_of_text|>"
            output_texts.append(text)
        return output_texts
    response_template = " ### Rewrite:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        args.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # dataset_text_field="text",
        formatting_func=formatting_prompts_func,
        packing=False,
        max_seq_length=args.max_seq_length,
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=collator
    )
    print("\n")
    trainer.model.print_trainable_parameters() # Print out the trainable parameters
    assert trainer.train_dataset.num_rows == len(train_data)
    print(tokenizer.decode(trainer.train_dataset[0]['input_ids']))

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument(
        '--output_dir', type=str, default="style_training/outputs")
    parser.add_argument(
        '--hf_token', type=str, default=None)
    parser.add_argument(
        '--lora_r', type=int, default=8)
    parser.add_argument(
        '--lora_alpha', type=int, default=32)
    parser.add_argument(
        '--lora_dropout', type=float, default=0.01)
    parser.add_argument(
        '--style', type=str, default='length_long')
    parser.add_argument(
        '--data_path', type=str, default=None)
    parser.add_argument(
        '--seed', type=int, default=0)
    parser.add_argument(
        '--save_ratio', type=int, default=4)
    parser.add_argument(
        '--epochs', type=int, default=5)
    parser.add_argument(
        '--train_batch_size', type=int, default=6)
    parser.add_argument(
        '--eval_batch_size', type=int, default=4)
    parser.add_argument(
        '--max_seq_length', type=int, default=512)
    main(parser.parse_args())
