"""
Trains a LoRA adapter on a decoder-only LM to rewrite text towards a specific style dimension
from the DiSC dataset (https://huggingface.co/datasets/hallisky/DiSC), such as 
more sarcasm, less sarcasm, more formality, etc.

Replace $HF_TOKEN with your huggingface token. You must have access to Meta-LLama-3-8B to run this.

# Example to train LLama-3-8B to be more sarcastic
CUDA_VISIBLE_DEVICES=0 python3 -m train_adapters.train_lora \
    --model meta-llama/Meta-Llama-3-8B \
    --hf_token $HF_TOKEN \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --data_path data_creation/generated_data_orig.json \
    --style sarcasm_more;

To train on other styles within DiSC, substitute "sarcasm_more" in the above command with one of the following styles

    sarcasm_less
    formality_informal
    formality_formal
    grade_elementary
    grade_highschool
    function_more
    function_less
    length_short
    length_long
    voice_active
    voice_passive
    type_persuasive
    type_expository
    type_narrative
    type_descriptive

Feel free to add logic to add your own parallel style data to train on as well
"""

import os
os.environ['TRANSFORMERS_CACHE'] = '../cache/'
os.environ['HF_HOME'] = '../cache/'
os.environ['HUGGINGFACE_HUB_CACHE'] = '../cache/'

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
import argparse
from huggingface_hub import login
import json
import sys
from sklearn.model_selection import train_test_split
from tokenizers.processors import TemplateProcessing

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
    
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    print(f"\n\t*\tModels are on {device}\n")

    # Initialize LoRA configuration for model adaptation    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        bias = "none")

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_eos_token=True)

    # Set up custom tokenization due to bug in LLama-3
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id), 
            (f"{eos}", tokenizer.eos_token_id)
        ],
    )
    # Check this is working - the last token should be eos
    assert tokenizer("Hey what's up with you I'm gonna go").input_ids[-1] == tokenizer.eos_token_id
    assert tokenizer("Hey what's up with you I'm gonna go", max_length=5, truncation=True).input_ids[-1] == tokenizer.eos_token_id

    if not tokenizer.pad_token: # Set pad token if it doesn't exist
        tokenizer.add_special_tokens({'pad_token': '<padding_token>'})

    """
    Three ways to load peft config:
    * AutoModelForCausalLM.from_pretrained(mmodel, peft_config = peft_config)
    * PeftModel.from_pretrained(model, peft_config)
    * peft.get_peft_model(model, peft_config)
    """

    steps = len(train_data)/(args.train_batch_size*torch.cuda.device_count())
    save_steps = math.ceil(steps / args.save_ratio) # Save every quarter epoch
    print(f"\n\t*\tSave steps is {save_steps}\n")

    # Set up training arguments
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

    # Define a function to format prompts for training and the response format
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['original'])):
            text = f"### Original: {example['original'][i]}\n ### Rewrite: {example['rewrite'][i]}"
            output_texts.append(text)
        return output_texts
    response_template = " ### Rewrite:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer)) # Resize to add pad token. Value doesn't matter
    trainer = SFTTrainer(
        model,
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

    trainer.train() # Train the model 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on a decoder-only language model (LM) to rewrite text towards a specific style dimension from the DiSC dataset.")

    parser.add_argument(
        '--model', 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B",
        help="Path to the pre-trained model or model identifier from Hugging Face hub."
    )

    parser.add_argument(
        '--output_dir', 
        type=str, 
        default="style_training/outputs",
        help="Directory to save the trained model and other outputs."
    )

    parser.add_argument(
        '--hf_token', 
        type=str, 
        default=None,
        help="Your Hugging Face token to access private models. Replace $HF_TOKEN with your token."
    )

    parser.add_argument(
        '--lora_r', 
        type=int, 
        default=8,
        help="LoRA rank, which defines the number of low-rank adaptations."
    )

    parser.add_argument(
        '--lora_alpha', 
        type=int, 
        default=32,
        help="LoRA alpha, a scaling factor for LoRA updates."
    )

    parser.add_argument(
        '--lora_dropout', 
        type=float, 
        default=0.01,
        help="Dropout rate for LoRA layers."
    )

    parser.add_argument(
        '--style', 
        type=str, 
        default='length_long',
        help="Style dimension to train the model on. Available styles include sarcasm_more, sarcasm_less, formality_informal, formality_formal, etc."
    )

    parser.add_argument(
        '--data_path', 
        type=str, 
        default=None,
        help="Path to the JSON file containing training data with original and rewritten texts for the specified style."
    )

    parser.add_argument(
        '--seed', 
        type=int, 
        default=0,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        '--save_ratio', 
        type=int, 
        default=4,
        help="Ratio of the total training steps to save checkpoints (e.g., 4 saves every quarter epoch)."
    )

    parser.add_argument(
        '--epochs', 
        type=int, 
        default=7,
        help="Number of training epochs."
    )

    parser.add_argument(
        '--train_batch_size', 
        type=int, 
        default=6,
        help="Batch size for training."
    )

    parser.add_argument(
        '--eval_batch_size', 
        type=int, 
        default=4,
        help="Batch size for evaluation."
    )

    parser.add_argument(
        '--max_seq_length', 
        type=int, 
        default=512,
        help="Maximum sequence length for the input text."
    )

    main(parser.parse_args())
