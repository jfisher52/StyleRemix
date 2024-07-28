"""
This file extracts n=500 paragraphs from a book, blog, and wiki dataset. 
They are then combined to create the base style train dataset used to create the style datasets for StyleMix
"""

from datasets import load_dataset
import numpy as np
from nltk import sent_tokenize
import random
import os
from tqdm import tqdm
import json
from unidecode import unidecode
import string
import re
import itertools

MAX_WC=150
MIN_WC=20
MAX_PARA=5
MIN_PARA=2

random.seed(0)
np.random.seed(0)

"""
Makes sublists randomly from list
"""
def split_list(input_list):
    output_list = []
    while input_list:
        if len(input_list) < MIN_PARA:
            sublist_length = len(input_list)
        else:
            sublist_length = random.randint(MIN_PARA, min(MAX_PARA, len(input_list)))
        output_list.append(input_list[:sublist_length])
        input_list = input_list[sublist_length:]
    return output_list

"""
Returns whether or not a paragraph is valid
"""
def validParagraph(paragraph):
    def isEnglish(s):
        return s.isascii()

    def word_count(s):
        """
        Returns the number of words in the string s
        """
        # Remove the punctuation
        s_nopunc = s.translate(str.maketrans('', '', string.punctuation))
        # Split by whitespace and return the length
        return len(s_nopunc.split())
    
    return (len(sent_tokenize(paragraph)) <= MAX_PARA) \
        and (len(sent_tokenize(paragraph)) >= MIN_PARA) \
        and (word_count(paragraph) <= MAX_WC) \
        and (word_count(paragraph) >= MIN_WC) \
        and isEnglish(paragraph)

cache_dir = "../cache/"
save_dir = "data_creation/style_train_data/"

n = 500 
book_data = load_dataset("kmfoda/booksum", cache_dir=cache_dir)
wiki_data = load_dataset("wikipedia", language="en", date="20220301", cache_dir=cache_dir)
blog_data = load_dataset("blog_authorship_corpus", cache_dir=cache_dir)

# Shuffle the data deterministically
train_book_data = book_data["train"].shuffle(seed=0)
train_wiki_data = wiki_data["train"].shuffle(seed=0)
train_blog_data = blog_data["train"].shuffle(seed=0)

# Randomly sample n=500 paragraph from books
# Training set n = 19200
if not os.path.exists(save_dir + "book_style_train_data.json"):
    book_train_data = []
    count = 0
    for idx in tqdm(range(5000, 10000)):
        chapter = train_book_data[idx]["chapter"] # extract data
        paragraphs = chapter.split("\n\n") # split paragraphs
        paragraphs = [re.sub(r'\s+', ' ', unidecode(p)).strip() for p in paragraphs]

        # filter for # sentences 2-5 and less than 300 words, more than 10 words
        filter_paragraphs = [paragraph for paragraph in paragraphs if validParagraph(paragraph)] 
        if len(filter_paragraphs) != 0:
            paragraph_sample = np.random.randint(0, len(filter_paragraphs)) # randomly select paragraph
            book_train_data.append({
                "book_id": train_book_data[idx]['book_id'], 
                "content": filter_paragraphs[paragraph_sample]
                })
            count +=1
        if count >= n:
            break
    with open(save_dir + "book_style_train_data.json", 'w') as f:
        json.dump(book_train_data, f, indent=4)
book_train_data = json.load(open(save_dir + "book_style_train_data.json", 'r'))

# Randomly sample n=500 paragraph from wiki
if not os.path.exists(save_dir + "wiki_style_train_data.json"):
    wiki_train_data = []
    count = 0

    for idx in tqdm(range(5000, 10000)):
        wiki = train_wiki_data[idx]["text"]

        # Don't want text after 'see also' or 'References
        for end_str in ["See also\n", "See also \n", "References\n", "References \n", "Bibliography\n", "Bibliography \n"]:
            wiki = wiki.split(end_str)[0]

        paragraphs = wiki.split("\n\n") # split paragraphs
        paragraphs = list(itertools.chain(*[p.split("\n") for p in paragraphs]))
        paragraphs = [re.sub(r'\s+', ' ', unidecode(p)).strip() for p in paragraphs]

        # filter for # sentences 2-5 and less than 200 words, more than 10 words
        filter_paragraphs = [paragraph for paragraph in paragraphs if validParagraph(paragraph)] 
        if len(filter_paragraphs) != 0:
            paragraph_sample = np.random.randint(0, len(filter_paragraphs)) # randomly select paragraph
            wiki_train_data.append({
                "title": train_wiki_data[idx]["title"],
                "content": filter_paragraphs[paragraph_sample]
                })
            count += 1
        
        if count >= n:
            break
    with open(save_dir + "wiki_style_train_data.json", "w") as f:
        json.dump(wiki_train_data, f, indent=4)
wiki_train_data = json.load(open(save_dir + "wiki_style_train_data.json", "r"))

# Randomly sample n=500 paragraph from blog
if not os.path.exists(save_dir + "blog_style_train_data.json"):
    blog_train_data = []
    count = 0
    for idx in tqdm(range(5000, 10000)):
        blog = train_blog_data[idx]['text'] # extract data
        blog = blog.replace("urlLink", "")

        # Make artificial paragraphs
        paragraphs = [' '.join(sp) for sp in split_list(sent_tokenize(blog))] 
        paragraphs = [re.sub(r'\s+', ' ', unidecode(p)).strip() for p in paragraphs]

        # filter for # sentences 2-5 and less than 300 words, more than 10 words
        filter_paragraphs = [paragraph for paragraph in paragraphs if validParagraph(paragraph)] 
        if len(filter_paragraphs) != 0:
            paragraph_sample = np.random.randint(0, len(filter_paragraphs)) # randomly select paragraph
            blog_train_data.append({
                "date": train_blog_data[idx]['date'], 
                "content": filter_paragraphs[paragraph_sample]
                })
            count +=1

        if count >= n:
            break
    with open(save_dir + "blog_style_train_data.json", "w") as f:
        json.dump(blog_train_data, f, indent=4)
blog_train_data = json.load(open(save_dir + "blog_style_train_data.json", "r"))

# Ensemble the data sources and save
book_data = [book_train_data[i]['content'] for i in range(n)]
wiki_data = [wiki_train_data[i]['content'] for i in range(n)]
blog_data = [blog_train_data[i]['content'] for i in range(n)]

train_data = {
    "book_data": book_train_data, 
    "wiki_data": wiki_train_data, 
    "blog_data": blog_train_data, 
    "mixture_data": book_data + wiki_data + blog_data
    }
with open(save_dir + "style_train_data_combined.json", "w") as f:
    json.dump(train_data, f, indent=4)
