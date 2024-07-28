"""
This file creates the 16-parallel style training datasets for StyleMix.
"""

from openai import OpenAI
import torch
import json
from tqdm import tqdm

# Create an assitant
OPENAI_API_KEY= "insert_openai_key"

# Create client with openAI key
client = OpenAI(api_key=OPENAI_API_KEY)

def model_generations(prompts, all_data):
    generation_dict = {}
    # cycle through all the prefix (different aspects of style )
    for prefix, prefix_name in tqdm(prompts, desc="Iterating through prompts"):
        generation_dict[prefix_name] = {}
        # cycle through all the datasets (blog, wiki, book)
        for dataset, dataset_name in tqdm(all_data, leave=False, desc = "iterating through data"):
            generations = []
            originals = []
            for data in tqdm(dataset, leave=False, desc="going through data"):
                completion = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{
                        "role": "user", 
                        "content": f"{prefix}\nParagraph: {data['content']} \nRewrite: "}
                    ],
                )
                generations.append(completion.choices[0].message.content)
                originals.append(data)
            generation_dict[prefix_name][dataset_name] = {"originals": originals, "generations": generations}
            torch.save(generation_dict, "data_creation/style_train_data_model_generation")
    return generation_dict

prompts = [
    ("Rewrite the following paragraph to include the same content but being more succint.", "length_short"), # shorter length
    ("Rewrite the following paragraph to include the same content but being more verbose.",  "length_long"),# longer length
    ("Rewrite the following paragraph to include the same content but using language an early elementary school student can understand.", "grade_elementary"), # elementary reading
    ("Rewrite the following paragraph to include the same content but use high school reading level or above.", "grade_highschool"), # high school reading
    ("Rewrite the following paragraph to include the same content but using far less function words (i.e. pronouns, determiners, and conjunctions).", "function_less"), # less function words
    ("Rewrite the following paragraph to include the same content but using far more function words (i.e. pronouns, determiners, and conjunctions).", "function_more"), # more function words
    ("Rewrite the following paragraph to include the same content but with more sarcasm.", "sarcasm_more"), # more sarcasm
    ("Rewrite the following paragraph to include the same content but with less sarcasm.", "sarcasm_less"), # less sarcasm
    ("Rewrite the following paragraph to include the same content but with more formal language.", "formality_formal"), # more formal
    ("Rewrite the following paragraph to include the same content but with more informal language.", "formality_informal"), # less formal (more informal)
    ("Rewrite the following paragraph to include the same content but with active voice.", "voice_active"), # active voice
    ("Rewrite the following paragraph to include the same content but with passive voice.", "voice_passive"), # passive voice
    ("Rewrite the following paragraph to include the same content but with persuasive writing style.", "type_persuasive"), # persuasive writing type
    ("Rewrite the following paragraph to include the same content but with expository writing style.", "type_expository"), # expository writing type
    ("Rewrite the following paragraph to include the same content but with narrative writing style.", "type_narrative"), # narrative writing type
    ("Rewrite the following paragraph to include the same content but with descriptive writing style.", "type_descriptive"), # descriptive writing type
    ] 

data = json.load(open("data_creation/style_train_data_combined.json", "r"))
all_data = [
    (data['book_data'], "book"),
    (data['wiki_data'], "wiki"),
    (data['blog_data'], "blog")
    ]
generations = model_generations(prompts, all_data)

torch.save(generations, "data_creation/style_train_data_model_generation") # save as torch file
with open("data_creation/style_train_data_model_generation.json", "w") as f: # save as .json
    json.dump(generations, f, indent=4)