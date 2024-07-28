"""
Main file that contains the different StyleRemix obfuscation methods
"""
import os
os.environ['TRANSFORMERS_CACHE'] = '../cache'
os.environ['HF_HOME'] = '../cache'
os.environ['HUGGINGFACE_HUB_CACH'] = '../cache'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse
from unidecode import unidecode
from src.eval_data import automatic_eval, load_eval_models, load_eval_class_model
from src.utils import identitify_styles_to_change_average, identitify_styles_to_change_one_each, create_ensemble_weight_dict, convert_to_format, clean_generations
from src.generate_lora_adapters import generate_lora_adapter, lorahub_generation, extract_lora_adapters
import numpy as np
import pandas as pd
import logging
import time
import sys

def main(args):
    exp_start_time = time.time()
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',  stream=sys.stdout)
    logging.info('Arguments: %s', args)
    
# Step 1. Download Data
    logging.info("Step 1: Download data")
    original_data = torch.load(args.input_path)
    
    # flatten data to make dictionary of authors
    new_dict = {}
    for outer_key, inner_dict in original_data.items():
        # Update the new dictionary with key-value pairs from the inner dictionary
        new_dict.update(inner_dict)
    original_data = new_dict
    
# Step 2. Run Automatic Evaluation (by author or by text)
    logging.info("Step 2: Run Automatic Evaluations to Choose Weights")
    if args.eval_test_average_dir != None:
        logging.info(f"Automatic Evaluation exists: {args.eval_test_average_dir}")
        average_original_eval_df = pd.DataFrame(torch.load(args.eval_test_average_dir))
        cola_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA', cache_dir = args.cache_dir).to(args.device)
        cola_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA', cache_dir = args.cache_dir, map_location = args.device)
        
    else:
        logging.info("Load necessary models")
        # Load Necessary Models
        sarcasm_model, sarcasm_tokenizer, voice_model, voice_tokenizer, type_model, type_tokenizer, formality_model, formality_tokenizer, cola_model, cola_tokenizer, sim_model = load_eval_models(args.device, args.cache_dir) 
    
        logging.info("Run automatic evaluations")
        original_eval = {}
        for author in list(original_data.keys()):
            eval_text = automatic_eval(original_data[author], args.device, obfuscation = False, author = str(author),\
                                                        sarcasm_model=sarcasm_model, sarcasm_tokenizer=sarcasm_tokenizer, formality_model=formality_model, formality_tokenizer = formality_tokenizer,\
                                                        voice_model = voice_model, voice_tokenizer = voice_tokenizer, type_model=type_model, type_tokenizer=type_tokenizer, \
                                                        cola_model=cola_model, cola_tokenizer=cola_tokenizer) 
            original_eval[author] = eval_text
        torch.save(original_eval, args.output_dir + "original_eval_df")
        
        # Calcuate average automatic evaluations over each author
        logging.info("Decide on which style to obfuscate")
        average_original_eval = {}
        for style in list(eval.keys()):
            average_original_eval[style] = {}
            for metric in list(eval[style]):
                if metric == "grade_level_dict":
                    average = np.nanmean(eval[style][metric]['fl'])
                else:
                    average = np.nanmean(eval[style][metric])
                average_original_eval[style][metric] = average
        average_original_eval_df = pd.DataFrame(average_original_eval)
        torch.save(average_original_eval_df, args.output_dir + "average_original_eval_df")
    
    # finalize style dataframe
    style_only_df = average_original_eval_df.T
    style_only_df = style_only_df[['words_per_sent', 'percent_function_words', 'grade_level_average', 
                                  'formal_conf', 'sarcasm_conf', 'voice_conf', 
                                  'type_persuasive_conf', 'type_narrative_conf', 'type_expository_conf', 'type_descriptive_conf']]
    style_only_df = style_only_df.T
    
    # Choose which style axes to change 
    if args.style_selector == "one_of_each":
        styles_to_change = identitify_styles_to_change_one_each(style_only_df, args.top_n_styles_to_change, args.weight_selection_method, shuffle=args.shuffle_style_selection)
    elif args.style_selector == "together":
        styles_to_change = identitify_styles_to_change_average(style_only_df, args.top_n_styles_to_change, args.weight_selection_method, shuffle=args.shuffle_style_selection) 
    elif args.style_selector == "random":
        styles_to_change = identitify_styles_to_change_average(style_only_df, args.top_n_styles_to_change, args.weight_selection_method, random_styles = True)
    else:
        print("An eligible style selector method was not choosen. Please choose from: 'one_of_each', 'together', or 'random'")

# Step 3. Make obfuscation decision based on each text individually
    logging.info(f"Step 3: Starting Obfuscation using {args.ensemble_method} ")
    # if you already started to obfuscate, do not overwrite!
    if os.path.exists(args.output_dir + "obf_generation_" + args.ensemble_method):
        obf_generation = torch.load(args.output_dir + "obf_generation_" + args.ensemble_method)
    else:
        obf_generation = {}
        
    for author in list(original_data.keys()):
        logging.info(f"Starting Obfuscation: {author}")
        if author in list(obf_generation.keys()): #don't re-do an author's generation
            continue
        
        if args.ensemble_method == "lorahub": # this is a new type of weight optimization technique
            prompts = [convert_to_format(unidecode(p)) for p in original_data[author]]
            lora_adapters_ls = extract_lora_adapters(styles_to_change[author]['styles'], styles_to_change[author]['direction'])
            # Use Lorahub to get optimized weights
            optimized_weights, adpater_ls = lorahub_generation(args.model, args.hf_token, lora_adapters_ls, prompts, args.lorahub_num_learning_examples, style_ls = styles_to_change[author], device = args.device, max_lora_inference_step = args.lorahub_max_inference_step, weight_only = True)
            ensemble_weights_dict = create_ensemble_weight_dict(author, styles_to_change[author], weights = optimized_weights)
            # Generate using LoRA adapters
            output = generate_lora_adapter(prompts, args.model, args.hf_token, ensemble_weights_dict , args.output_dir, batch_size = args.batch_size, merge_model_type=args.combination_type)
            obf_generation[author] = {'originals': original_data[author], 'generations': output, 'weights':optimized_weights, 'styles': styles_to_change[author]}
            torch.save(obf_generation, args.output_dir + "obf_generation_" + args.ensemble_method)
        
        elif args.ensemble_method == "sequential":
            # cycle through each style and create a generation (type of iterative changing)
            ensemble_weights_dict = create_ensemble_weight_dict(author, styles_to_change[author], sequential=True)
            for j, style in enumerate(ensemble_weights_dict[author]):
                if j == 0: #start with original prompts
                    prompts = original_data[author]
                    prompts = [convert_to_format(unidecode(p)) for p in prompts]
                    obf_generation[author] = {}
                    obf_generation[author]['originals'] = original_data[author]
                generations_raw = generate_lora_adapter(prompts, args.model, args.hf_token, {author: ensemble_weights_dict[author][j]}, args.output_dir, batch_size = args.batch_size, merge_model_type=args.combination_type)[author]['outputs']
                # if empty or low grammar than replace with one before:
                generations_clean, grammar_scores, grammar_mask, no_text_mask = clean_generations(generations_raw, prompts, cola_model, cola_tokenizer, args.device, grammar_threshold = args.grammar_threshold)
                obf_generation[author]["step_" + str(j)] = {'generations_clean': generations_clean,'generations_raw': generations_raw, "grammar_scores":grammar_scores, "grammar_mask":grammar_mask, "no_text_mask": no_text_mask, 'weights':ensemble_weights_dict, 'style_change': styles_to_change[author]['styles'][j] + "_ "+ styles_to_change[author]['direction'][j]}
                torch.save(obf_generation, args.output_dir + "obf_generation_" + args.ensemble_method)
                prompts = [convert_to_format(unidecode(g)) for g in generations_clean]
        
        else:
            ensemble_weights_dict = create_ensemble_weight_dict(author, styles_to_change[author])
            prompts = original_data[author]
            prompts = [convert_to_format(unidecode(p)) for p in prompts]
            output = generate_lora_adapter(prompts, args.model, args.hf_token, ensemble_weights_dict, args.output_dir, batch_size = args.batch_size, merge_model_type=args.combination_type)
            obf_generation[author] = {'originals': original_data[author], 'generations': output[author]['outputs'], 'weights':ensemble_weights_dict, 'styles': styles_to_change}
            torch.save(obf_generation, args.output_dir + "obf_generation_" + args.ensemble_method)
            
    save_name = args.output_dir + "obf_generation_" + args.ensemble_method
    logging.info(f"Saved Generations: {save_name}")
    logging.info(f"Total Time for Generations: {time.time - exp_start_time}")
    
# Step 4. Evluate Generations
    generations_df = torch.load(args.output_dir + "obf_generation_" + args.ensemble_metho)
    for author in generations_df:
        prompts = original_data[author]
        sarcasm_model, sarcasm_tokenizer, voice_model, voice_tokenizer, type_model, type_tokenizer, formality_model, formality_tokenizer, cola_model, cola_tokenizer, sim_model = load_eval_models(args.device, args.cache_dir) 
        classifier = load_eval_class_model(author, args.device, args.cache_dir)
        obf_generation[author]['auto_eval'] = automatic_eval(original_data[author], args.device, obfuscation = True, author = str(author),\
                                            sarcasm_model=sarcasm_model, sarcasm_tokenizer=sarcasm_tokenizer, formality_model=formality_model, formality_tokenizer = formality_tokenizer,\
                                            voice_model = voice_model, voice_tokenizer = voice_tokenizer, type_model=type_model, type_tokenizer=type_tokenizer, \
                                            cola_model=cola_model, cola_tokenizer=cola_tokenizer, classifier= classifier, sim_model= sim_model) 

        torch.save(obf_generation, args.output_dir + "obf_evaluations_"+ args.ensemble_method)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parameters and Permissions
    parser.add_argument(
        '--hf_token', type=str, default=None, help="Huggingface login. Needed if base model requires a huggingface login")
    parser.add_argument(
        '--device', type=str, default="cuda:0", help="Which GPU to use (needed)")
    
    # Data Directories and Models
    parser.add_argument(
        '--model', type=str, default="meta-llama/Meta-Llama-3-8B", help="Base model used for generation")
    parser.add_argument(
        '--input_path', type=str, default="../data/test_data/StyleMix", help="Directory of test data (torch version)")
    parser.add_argument(
        '--cache_dir', type=str, default="../cache", help="Cache directory used for loading models")
    parser.add_argument(
        '--output_dir', type=str, default="../results", help="Exact directory/name of output file for final results")
    parser.add_argument(
        '--eval_test_average_dir', type=str, default="../data/test_data/StyleMix_averages_by_author", help="If already calculated then put directory, if not than put 'None'")

    # LoRA/LoRAHub Parameters
    parser.add_argument(
        '--combination_type', type=str, default='cat', help="Type of LoRA adapter combination. For full list see https://huggingface.co/blog/peft_merging")
    parser.add_argument(
        '--batch_size', type=int, default=1)
    parser.add_argument(
        '--lorahub_num_learning_examples', type = int, default = 5)
    parser.add_argument(
        '--lorahub_max_inference_step', type = int, default = 3
    )
    
    # Obfuscation Method Parameters
    parser.add_argument(
    '--top_n_styles_to_change', type = int, default = 5, help="Number of style axes to change (up to 7)")

    parser.add_argument(
        '--ensemble_method', type = str, default = "lorahub", help="Type of ensemble method. Choices: 'lorabub' or 'sequential'"
    )
    parser.add_argument(
        '--style_selector', type = str, default = "together", help="How to select styles to change. Choices: 'one_of_each', 'together', 'random'"
    )
    parser.add_argument(
        '--shuffle_style_selection', type = bool, default = True, help="Whether to shuffle the styles or keep them in the order of most different to least'"
    )
    parser.add_argument(
        '--weight_selection_method', type = str, default = "threshold", help="How to select the weight for each style. Choices: 'threshold', 'n_std', "
    )
    parser.add_argument(
        '--grammar_threshold', type=float, default=0.5, help="This creates a cut-off grammar for the sequential method")
    args = parser.parse_args()
    main(args)
