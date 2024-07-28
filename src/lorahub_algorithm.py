"""
This file supports using LoraHub for the generation.
The code was adapted from https://github.com/sail-sg/lorahub
"""
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import numpy
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import PeftModel, PeftConfig
from functools import partial
from typing import List, Optional
import copy
from huggingface_hub import login
from src.eval_data import eval_length, eval_function_words, eval_grade_level, eval_formality, eval_sarcasm, eval_type, eval_voice, eval_grammar

def load_base_model_and_lora_modules(lora_module_list: List[str], model_name_or_path: Optional[str] = None, hf_token = None, cache_dir = "../cache"):
    """load base model and lora modules from huggingface model hub

    Args:
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
        model_name_or_path (Optional[str]): base model name, default is None
    """
    # Set the huggingface token
    if hf_token:
        login(hf_token)
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load basic model
    default_peft_model_id = lora_module_list[0]
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path
        
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir = cache_dir)
    # Reinitialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, add_bos_token=True, add_eos_token=False,padding_side="left", cache_dir = cache_dir)
    tokenizer.add_special_tokens({'pad_token': '<padding_token>'})
    
    model = AutoModelForCausalLM.from_pretrained(model, cache_dir = cache_dir).to('cuda').eval()
    model.resize_token_embeddings(len(tokenizer)) # Resize to add pad token. Value doesn't matter
       
    # 0 is the default model
    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id, cache_dir = cache_dir)
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
               
    return peft_model, tokenizer, cache


def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets

def lora_loss(example_dataset, model, tokenizer, batch_size, device, style_ls=None, cache_dir = "../cache"):
    inputs = example_dataset['input']
    outputs_ls = []
    with torch.no_grad():
        for batch_idx in range(0, len(inputs), batch_size):
            cur_text = inputs[batch_idx:batch_idx + batch_size]
            prompts = tokenizer(cur_text, return_tensors="pt", padding=True, max_length=256, truncation=True)  # , add_special_tokens=False)
            prompts = {k: v.to("cuda") for k, v in prompts .items()}
            input_length = prompts["input_ids"].shape[1]

            outputs = model.generate(
                **prompts,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs_ls.extend([o.split("###")[0].strip() for o in tokenizer.batch_decode(outputs[:, input_length:],  skip_special_tokens=True)])  
    styles = style_ls['styles']
    directions = style_ls['direction']
    loss_ls = []
    for style, direction in zip(styles, directions):
        if style == "n_words":
            orig_n_words, _, _ = eval_length(inputs) # original length
            n_words, n_characters, char_per_word = eval_length(outputs_ls) #obfuscated length
            value = [obf/orig for obf, orig in zip(n_words, orig_n_words)]
            
        elif style == "percent_function_words":
            n_function_words, percent_function_words = eval_function_words(outputs_ls)
            value = percent_function_words
            
        elif style == "grade_level_average":
            grade_level_dict, grade_level_average = eval_grade_level(outputs_ls)
            value = [grade/15 for grade in grade_level_average]
            
        elif style == "percent_misspelled":
            n_misspelled, percent_misspelled, misspelled_words = eval_spelling(outputs_ls)
            value = percent_misspelled

        elif style == "formal_conf":
            formality_tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker", cache_dir = cache_dir, map_location = device)
            formality_model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker", cache_dir = cache_dir).to(device)
            formal_pred, formal_conf = eval_formality(outputs_ls, device, formality_model, formality_tokenizer)
            value = formal_conf

        elif style == "sarcasm_conf":
            sarcasm_tokenizer = AutoTokenizer.from_pretrained("hallisky/sarcasm-classifier-gpt4-data", cache_dir = cache_dir, map_location = device)
            sarcasm_model = AutoModelForSequenceClassification.from_pretrained("hallisky/sarcasm-classifier-gpt4-data", cache_dir = cache_dir).to(device)
            sarcasm_pred, sarcasm_conf = eval_sarcasm(outputs_ls, device, sarcasm_model, sarcasm_tokenizer)
            value = sarcasm_conf

        elif style == "voice_conf":
            voice_tokenizer = AutoTokenizer.from_pretrained("hallisky/voice-classifier-gpt4-data", cache_dir = cache_dir, map_location = device)
            voice_model = AutoModelForSequenceClassification.from_pretrained("hallisky/voice-classifier-gpt4-data", cache_dir = cache_dir).to(device)
            voice_pred, voice_conf = eval_voice(outputs_ls, device, voice_model, voice_tokenizer, batch_size = 8)
            value = voice_conf
        
        else:
            print("Metric not found to create loss!")

        # Logic: if "direction" is "lower", then the style was low and we not want it higher. Therefore, we give it a negative loss, to encourage increasing it (making it more negative)
        if direction == "lower":
            loss_ls.append(-1 * np.mean(value))
        else:
            loss_ls.append(np.mean(value))
        # if "type_conf" in style_ls:
            # type_pred, type_conf = eval_voice(inputs, device, type_model, type_tokenizer, batch_size = 8)
    # Want to include a grammar score        
    cola_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA', cache_dir = cache_dir).to(device)
    cola_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA', cache_dir = cache_dir, map_location = device)  
    grammar_score = eval_grammar(outputs_ls, cola_tokenizer, cola_model, device)
    loss_ls.append(-1* np.mean(grammar_score)) # want to have a high grammar score
 
    return(np.sum(loss_ls))

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares

def get_score(weights, model, tokenizer, cache, example_dataset, batch_size, get_loss, get_regular, device = None, styles = None):
    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # reload the model with the new adapter config
    set_peft_model_state_dict(model, final_state_dict)
        
    # minimize the metric
    loss = get_loss(example_dataset, model, tokenizer, batch_size, device, styles)
    # L1 regularization term
    metric_val = loss + get_regular(weights)
    
    return metric_val

def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict
    
def lorahub_learning(lora_module_list: List[str], 
                     example_inputs: List[str], 
                     max_inference_step: int,
                     example_outputs: None, 
                     model_name_or_path=None,
                     batch_size=None,
                     get_loss=lora_loss, 
                     get_regular=default_l1_regularization,
                     seed=42,
                     hf_token=None, 
                     style_ls = None,
                     device = None):
    # set seed for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    # load model
    model, tokenizer, cache = load_base_model_and_lora_modules(lora_module_list, model_name_or_path, hf_token)
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer) 
    get_score_partial = partial(get_score, 
                                model=model,
                                tokenizer=tokenizer, 
                                cache=cache,
                                example_dataset=dataset,
                                batch_size=batch_size,
                                get_loss=get_loss, 
                                get_regular=get_regular, 
                                device=device,
                                styles = style_ls)
    # set up the limit of the weights
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[0] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    print("> Begin to perform gradient-free optimization ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    # set the final weights
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()
    return recommendation.value, model, tokenizer