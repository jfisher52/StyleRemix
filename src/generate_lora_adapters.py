"""
This file creates the generations using LoRA adpaters (both with and without Lorahub)
"""
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from src.lorahub_algorithm import lorahub_learning, lora_loss

default_key = 'length_long'

## ------------ HELPER FUNCTIONS ------------##

# Creates format for LoRA adapters (same used during training)
def convert_to_format(text):
    return f"### Original: {text}\n ### Rewrite:"
# Process raw output to extract only the generation
def process_output(raw_output):
    # get rid the common pattern ### Rewrite: ###
    output = [output if output[0:4] != " ###" else output[4:] for output in raw_output]
    return([o.split("###")[0].strip() for o in output])

# Download pre-trained LoRA adapters
def lora_models():
    models = {
    "length_long": "hallisky/lora-length-long-llama-3-8b", #0
    "length_short": "hallisky/lora-length-short-llama-3-8b", #1
    "voice_active": "hallisky/lora-voice-active-llama-3-8b", #2
    "voice_passive": "hallisky/lora-voice-passive-llama-3-8b", #3
    'function_more': "hallisky/lora-function-more-llama-3-8b", #4
    "function_less": "hallisky/lora-function-less-llama-3-8b", #5
    "formality_formal": "hallisky/lora-formality-formal-llama-3-8b", #6
    "formality_informal": "hallisky/lora-formality-informal-llama-3-8b", #7
    "sarcasm_more": "hallisky/lora-sarcasm-more-llama-3-8b", #8
    "sarcasm_less": "hallisky/lora-sarcasm-less-llama-3-8b", #9
    "grade_highschool": "hallisky/lora-grade-highschool-llama-3-8b", #10
    "grade_elementary": "hallisky/lora-grade-elementary-llama-3-8b", #11
    "type_persuasive": "hallisky/lora-type-persuasive-llama-3-8b", #12
    "type_expository": "hallisky/lora-type-expository-llama-3-8b", #13
    "type_narrative": "hallisky/lora-type-narrative-llama-3-8b", #14
    "type_descriptive": "hallisky/lora-type-descriptive-llama-3-8b" # 15
    }
    static_style_axis_ls = ["words_per_sent_higher", "words_per_sent_lower","voice_conf_higher", "voice_conf_lower","percent_function_words_higher", "percent_function_words_lower", 
                            "formal_conf_higher","formal_conf_lower", "sarcasm_conf_higher", "sarcasm_conf_lower", 
                            "grade_level_average_higher","grade_level_average_lower", "type_persuasive_conf", "type_expository_conf", "type_narrative_conf", "type_descriptive_conf"]
    
    static_adapter_ls = models.keys() 
    
    return models, static_style_axis_ls, static_adapter_ls

def opposite_direction(direction):
        if direction == "higher":
            return("lower")
        else:
            return("higher")


# Extrac the LoRA adapters based on which style axes the user wants to change
def extract_lora_adapters(style_ls, direction_ls):
    models, static_style_axis_ls, static_adapter_ls = lora_models()
    adapter_ls = []
    for style, direction in zip(style_ls, direction_ls):
        opposite_dir = opposite_direction(direction)
        if "old_type" in style:
            style_plus_direction = "_".join(style.split("_")[-3:])
        else:
            style_plus_direction = style + "_" + opposite_dir
        adapter_index = static_style_axis_ls.index(style_plus_direction)
        adapter_name = list(models.keys())[adapter_index]
        adapter_ls.append(models[adapter_name])
    return(adapter_ls)

## ------------ GENERATION WITH LORA ADAPTERS + BASE MODEL ------------##

# Combine choosen LoRA adapter with base model
def generate_lora_adapter(prompts, model, hf_token, ensemble_weights_dict, output_dir, batch_size = 64, merge_model_type="cat", cache_dir = "../cache"):
    # Pre-process data
    text = [convert_to_format(p) for p in prompts]

    # Load pre-trained model adapters
    models, _, _ = lora_models()
    
    # Set the huggingface token
    if hf_token:
        login(hf_token)

    # Reinitialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model, add_bos_token=True, add_eos_token=False, padding_side="left", cache_dir = cache_dir)
    tokenizer.add_special_tokens({'pad_token': '<padding_token>'})
    
    model = AutoModelForCausalLM.from_pretrained(model, cache_dir = cache_dir).to('cuda').eval()
    model.resize_token_embeddings(len(tokenizer)) # Resize to add pad token. Value doesn't matter
    model = PeftModel.from_pretrained(model, models[default_key], adapter_name=default_key, cache_dir = cache_dir)
    for k in models:
        if k != default_key:
            _ = model.load_adapter(models[k], adapter_name=k)

    # Load adapters
    adapters = list(models.keys())
    for e in ensemble_weights_dict:
        model.add_weighted_adapter(
            adapters, 
            ensemble_weights_dict[e], 
            e, 
            combination_type=merge_model_type)
        
    all_outputs = {}
    if os.path.isfile(output_dir + "_generations"): #load generations if we have already loaded them
        all_outputs = torch.load(output_dir + "_generations")

    torch.no_grad()
    for adapter in ensemble_weights_dict:
        if adapter in list(all_outputs.keys()): #skip adapters we have already generated
            continue
        all_outputs[adapter] = {"outputs": []}

        # Running on the specified adapater
        print(f"\n\t*\tRunning on adapter {adapter}\n")
        model.set_adapter(adapter) #original name

        for batch_idx in range(0, len(text), batch_size):
            cur_text = text[batch_idx:batch_idx + batch_size]
            inputs = tokenizer(cur_text, return_tensors="pt", padding=True, max_length=256, truncation=True)  # , add_special_tokens=False)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
            output_raw = tokenizer.batch_decode(outputs[:, input_length:],  skip_special_tokens=True)
            processed_output = process_output(output_raw)
            all_outputs[adapter]["outputs"].extend(processed_output)
    return(all_outputs)

## ------------ HELPER FUNCTIONS FOR LORAHUB ------------##
# Code adapted from https://github.com/sail-sg/lorahub

# Function for LoraHub Generation
def get_examples_for_learning(eval_dataset, num_learning_examples):
    """
    Get a few examples to learn to compose given LoRA modules
    """
    prompts = []
    for prompt in eval_dataset[0:num_learning_examples]:
        prompts.append({"input": prompt})
    return(prompts)

def get_examples_for_inference(eval_dataset):
    """
    Inference on the examples to get the performance of the composed LoRA modules
    """
    prompts = []
    for prompt in eval_dataset:
        prompts.append({"input": prompt})
    return(prompts)

def get_lora_module_list(ensemble_weights_dict):
    """
    You can have a custom filtering strategy to select the modules to be used in the composition.
    """
    return ensemble_weights_dict


def lorahub_generation(model, hf_token, ensemble_weights_dict, eval_dataset, num_learning_examples, style_ls = None, max_lora_inference_step=10, batch_size = 16, device = None, cache_dir = "../cache_dir", weight_only = False):
    """
    Perform lorahub learning
    """
    # get a list of modules to be used in the composition
    modules = get_lora_module_list(ensemble_weights_dict)
    print("modules:", modules)

    # construct input list and output list
    example_inputs = []
    for example in get_examples_for_learning(eval_dataset, num_learning_examples):
        example_inputs.append(example["input"])

    # perform LoRAHub learning
    module_weights, model, tokenizer = lorahub_learning(model_name_or_path=model,
                                                        lora_module_list=modules,
                                                        example_inputs=example_inputs,
                                                        example_outputs = None,
                                                        max_inference_step=max_lora_inference_step,
                                                        batch_size=batch_size, 
                                                        get_loss = lora_loss,
                                                        hf_token=hf_token, 
                                                        style_ls=style_ls,
                                                        device = device)
    if weight_only:
        return(module_weights, modules)
    print("module_weights:", module_weights)

    """
    Perform inference to get predictions
    """
    # now you can use the model to perform inference
    example_inputs = []
    for example in get_examples_for_inference(eval_dataset):
        example_inputs.append(example["input"])


    example_predictions, perf = lorahub_inference(example_inputs=example_inputs,
                                                  model_or_name_path=model,
                                                  tokenizer_or_tokenizer_path=tokenizer,
                                                  batch_size= batch_size)

    # combine input_length_ls into one list
    return(example_predictions, perf, module_weights)