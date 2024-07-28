"""
This file supports obfuscate.py.
"""
import numpy as np
import re 
from unidecode import unidecode
from src.generate_lora_adapters import lora_models
from src.eval_data import eval_grammar
import random 


random.seed(0)
    
## --------- TEXT RELATED HELPER FUNCTIONS --------- ##
def clean_text(text):
    text = add_space_after_period(text)
    text = unidecode(text)
    return text

def add_space_after_period(text):
    pattern = r'\.([A-Z])'
    replacement = r'. \1'
    corrected_text = re.sub(pattern, replacement, text)
    return corrected_text

def convert_to_format(text):
    return f"### Original: {text}\n ### Rewrite:"

def normalize_row(row):
    return row / row.max()

def clean_generations(generations, originals, cola_model, cola_tokenizer, device, grammar_threshold = 0.8):
    # replace emtpy text with "No Generation" as a placeholder for the grammar eval
    generations = ["No Generation" if text == '' else text for text in generations] 
    grammar_scores = eval_grammar(generations, cola_tokenizer, cola_model, device)
    # Create a boolean mask where scores are above the threshold and where generations are empty
    grammar_mask = np.array(grammar_scores) > grammar_threshold
    no_text_mask = np.array(generations) == "No Generation"

    # Use the mask to select from generations and originals
    final_generations = np.where(grammar_mask, generations, originals)
    final_generations = np.where(no_text_mask, originals, generations)

    # Convert final_generations to a list 
    return(final_generations.tolist(), grammar_scores, grammar_mask, no_text_mask)

## --------- CHOOSING LORA ADAPTER WEIGHTS HELPER FUNCTIONS --------- ##
def group_by_classifier(df, author):
    speeches = ['trump', 'obama', 'bush']
    novels = ["hemingway", "fitzgerald", "woolf"]
    amt = ["amt_h", "amt_pp", "amt_qq"]
    blog = ["blog_5_1", "blog_5_2", "blog_5_3", "blog_5_4", "blog_5_5"]
    if author in speeches:
        return(df[speeches])
    elif author in novels:
        return(df[novels])
    elif author in amt:
        return(df[amt])
    elif author in blog:
        return(df[blog])
    else:
        print("Author is not in any classifier!")
        
def shuffle_dict_in_sync(data):
    # Extract the keys and lists
    keys = list(data.keys())
    values = list(data.values())
    # Combine/shuffle the lists into a list of tuples
    combined = list(zip(*values))
    random.shuffle(combined)
    
    # Unzip the combined list back into individual lists
    shuffled_values = list(zip(*combined))
    shuffled_dict = {keys[i]: list(shuffled_values[i]) for i in range(len(keys))}
    return shuffled_dict

def choose_weights(author_val, style_std, selection_type = "n_std"):
    if selection_type == "n_std":
        return author_val/style_std
    elif selection_type == "threshold":
        n_std = author_val/style_std
        if n_std <= 1:
            return(.7)
        elif n_std>1 and n_std<=2:
            return(.9)
        elif n_std>2 and n_std<=3:
            return(1.2)
        else:
            return(1.5)
    else:
        print("Weight selection method is not available")
        
def identitify_styles_to_change_one_each(df, top_n, magnitude_selection, shuffle = False):
    styles_to_change = {}
    for author in df.columns:
        # Calculate average writing style vector of all other authors in their classifier!
        classifier_specific_df = group_by_classifier(df, author)
        normalized_df = classifier_specific_df.apply(normalize_row, axis=1)
        
        # Split into to groups of style characteristics
        extrinsic_styles_normalized_df = normalized_df.loc[["words_per_sent", "percent_function_words", "grade_level_average"]]
        intrinsic_styles_normalized_df = normalized_df.loc[["sarcasm_conf", "formal_conf", "voice_conf", "type_narrative_conf", "type_persuasive_conf", "type_expository_conf", "type_descriptive_conf"]]
        
        # Calculate average writing style vector of all other authors
        avg_extr_style = extrinsic_styles_normalized_df.mean(axis=1)
        std_extr_style = extrinsic_styles_normalized_df.std(axis=1)
        avg_intr_style = intrinsic_styles_normalized_df.mean(axis=1)
        std_intr_style = intrinsic_styles_normalized_df.std(axis=1)
        direction = []
        magnitude = []
        styles_ls = []
        type = 0
        for style_group_data, style_type in zip([extrinsic_styles_normalized_df, intrinsic_styles_normalized_df], ["extrinsic", "intrinsic"]):
            # Get author data and correct avg/std data
            author_vec = style_group_data[author]
            if style_type == "extrinsic":
                avg_style = avg_extr_style
                std_style = std_extr_style
            else:
                avg_style = avg_intr_style
                std_style = std_intr_style
            # Find difference between author data and average data
            diff_vec = author_vec - avg_style
            # Select the biggest difference as styles to change
            val = [np.abs(x) for x in diff_vec]
            sorted_indices = sorted(range(len(val)), key=lambda i: val[i], reverse=True)
            count = 0 # count for top_n styles to extract
            # Cycle through and extract the style name and magnitude (# of std from average)
            for index in sorted_indices: # add one in case there is a percent misspelled
                            # dealing with "type" values
                if diff_vec.index[index] in ['type_persuasive_conf', "type_narrative_conf", "type_expository_conf", "type_descriptive_conf"]:
                    if  type == 0:
                        # randomly choose any type that was not the original
                        new_type = random.choice([item for item in ['type_persuasive_conf', "type_narrative_conf", "type_expository_conf", "type_descriptive_conf"] if item != diff_vec.index[index]])
                        styles_ls.append("old_" + diff_vec.index[index] + "_new_" + new_type)
                        magnitude.append(choose_weights(val[index], std_style[index], selection_type=magnitude_selection))
                        direction.append("new_type")
                        type = 1 # can only choose 1 type
                        count += 1
                    else:
                        continue
                else:
                    if author_vec[index] > avg_style[index]:
                        direction.append("higher")
                        count += 1
                    else:
                        direction.append("lower")
                        count += 1
                    magnitude.append(choose_weights(val[index], std_style[index], selection_type=magnitude_selection))
                    styles_ls.append(diff_vec.index[index])
                if count == top_n:
                    break
        styles_to_change[author] = {"styles": styles_ls, "direction": direction, "magnitude": magnitude}
        if shuffle:
            styles_to_change[author] = shuffle_dict_in_sync(styles_to_change[author])
    return(styles_to_change)


def identitify_styles_to_change_average(df, top_n, magnitude_selection, random_styles = False, shuffle = False):
    styles_to_change = {}
    for author in df.columns:
        # Calculate average writing style vector of all other authors in their classifier!
        classifier_specific_df = group_by_classifier(df, author)
        normalized_df = classifier_specific_df.apply(normalize_row, axis=1)
        avg_style = normalized_df.mean(axis=1)
        std_style = normalized_df.std(axis=1)
        author_vec = normalized_df[author]
        diff_vec = author_vec - avg_style
        val = [np.abs(x) for x in diff_vec]
        if random_styles:
            sorted_indices = list(range(df.shape[0]))
            random.shuffle(sorted_indices)
        else:
            sorted_indices = sorted(range(len(val)), key=lambda i: val[i], reverse=True)
        direction = []
        magnitude = []
        styles_ls = []
        type = 0 #no type selected
        for index in sorted_indices: 
            # dealing with "type" values
            if diff_vec.index[index] in ['type_persuasive_conf', "type_narrative_conf", "type_expository_conf", "type_descriptive_conf"]:
                if  type == 0:
                    # randomly choose any type that was not the original
                    new_type = random.choice([item for item in ['type_persuasive_conf', "type_narrative_conf", "type_expository_conf", "type_descriptive_conf"] if item != diff_vec.index[index]])
                    styles_ls.append("old_" + diff_vec.index[index] + "_new_" + new_type)
                    magnitude.append(choose_weights(val[index], std_style[index], selection_type=magnitude_selection))
                    direction.append("new_type")
                    type = 1 # can only choose 1 type
                    if len(direction) == top_n:
                        break
                    continue
                else:
                    continue
            else:
                if author_vec[index] > avg_style[index]:
                    direction.append("higher")
                else:
                    direction.append("lower")
                magnitude.append(choose_weights(val[index], std_style[index], selection_type=magnitude_selection))
                styles_ls.append(diff_vec.index[index])
            if len(direction) == top_n:
                break
        styles_to_change[author] = {"styles": styles_ls, "direction": direction, "magnitude": magnitude}
        if shuffle:
            styles_to_change[author] = shuffle_dict_in_sync(styles_to_change[author])
    return(styles_to_change)

def opposite_direction(direction):
        if direction == "higher":
            return("lower")
        else:
            return("higher")


def create_ensemble_weight_dict(author, styles_to_change_df, weights = None, sequential = False, threshold_magnitude=1.5):
    _, static_style_axis_ls, static_adapter_ls = lora_models() 
    if weights == None:
        weights = styles_to_change_df['magnitude']
    if sequential:
        adapter_weight = [[0] * len(static_adapter_ls) for i in range(len(styles_to_change_df['styles']))]
    else:
        adapter_weight = [0] * len(static_adapter_ls)
    magnitude_index = 0
    weight_index = 0
    for style in list(styles_to_change_df['styles']):
        if "old_type" in style:
            style_plus_direction = "_".join(style.split("_")[-3:])
        else:
            style_plus_direction = style + "_" + opposite_direction(styles_to_change_df['direction'][magnitude_index])
        if style_plus_direction in static_style_axis_ls:
            style_index = static_style_axis_ls.index(style_plus_direction)
            if np.abs(weights[magnitude_index]) > threshold_magnitude: # if set magnitude is too large than cut it
                magnitude_cut = np.sign(weights[magnitude_index]) * 1.5 # keep the sign of the magnitude
            else:
                magnitude_cut = weights[magnitude_index]
            if sequential:
                adapter_weight[weight_index][style_index] = np.round(magnitude_cut,2)
            else:
                adapter_weight[style_index] = np.round(magnitude_cut,2)
            weight_index +=1
            magnitude_index += 1
    return {author: adapter_weight}
       
