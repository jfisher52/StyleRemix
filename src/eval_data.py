"""
This file calculates all the automatic evaluations used to select the LoRA adapters/weights.
"""
import numpy as np
import textstat
import string
import torch
from nltk import sent_tokenize
from nltk import pos_tag
import nltk
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
nltk.download('averaged_perceptron_tagger')
from torch.nn import CosineSimilarity
cos = CosineSimilarity(dim=-1)

# Load Necessary Models for Automatic Eval
def load_eval_models(device, cache_dir):
    # Sarcasm Classifier
    sarcasm_tokenizer = AutoTokenizer.from_pretrained("hallisky/sarcasm-classifier-gpt4-data", cache_dir = cache_dir, map_location = device)
    sarcasm_model = AutoModelForSequenceClassification.from_pretrained("hallisky/sarcasm-classifier-gpt4-data", cache_dir = cache_dir).to(device)
    
    # Voice Classifier
    voice_tokenizer = AutoTokenizer.from_pretrained("hallisky/voice-classifier-gpt4-data", cache_dir = cache_dir, map_location = device)
    voice_model = AutoModelForSequenceClassification.from_pretrained("hallisky/voice-classifier-gpt4-data", cache_dir = cache_dir).to(device)

    # Writing Type (Intent) Classifier
    type_tokenizer = AutoTokenizer.from_pretrained("hallisky/type-classifier-gpt4-data", cache_dir = cache_dir, map_location = device)
    type_model = AutoModelForSequenceClassification.from_pretrained("hallisky/type-classifier-gpt4-data", cache_dir = cache_dir).to(device)

    # Formality Classifier
    formality_tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta-base-formality-ranker", cache_dir = cache_dir, map_location = device)
    formality_model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta-base-formality-ranker", cache_dir = cache_dir).to(device)
    
    # Grammaticality Classifier
    cola_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA', cache_dir = cache_dir).to(device)
    cola_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA', cache_dir = cache_dir, map_location = device)
    
    # Content Preservation Model Helper
    sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    
    return(sarcasm_model, sarcasm_tokenizer, voice_model, voice_tokenizer, type_model, type_tokenizer, formality_model, 
           formality_tokenizer, cola_model, cola_tokenizer, sim_model)

# Load correct Authorship Classification Model
def load_eval_class_model(author, device, cache_dir):
    if author in ["hemingway", "wolf", "woolf", "fitzgerald"]:
        style = "novels"
    elif author in ["trump", "obama", "bush"]:
        style = "speeches"
    elif "amt" in author:
        style = "amt"
    elif "blog" in author:
        style = "blog"

    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir
    # Novel author
    if style == "novels":
        classifier = pipeline('text-classification', model="hallisky/author-classifier-roberta-large", tokenizer="roberta-large", top_k=None, function_to_apply='sigmoid', device=0, truncation = True, model_kwargs = {"cache_dir":cache_dir})
    # Speeches author
    elif style == "speeches":
        classifier = pipeline('text-classification', model="hallisky/speech-classifier-roberta-large", tokenizer="roberta-large", top_k=None, function_to_apply='sigmoid', device=device, truncation = True, model_kwargs = {"cache_dir":cache_dir})
    # AMT author
    elif style == "amt":
        classifier = pipeline('text-classification', model="hallisky/amt-classifier-roberta-large", tokenizer="roberta-large", top_k=None, function_to_apply='sigmoid', device=device, truncation = True, model_kwargs = {"cache_dir":cache_dir})
    # Blog author
    elif style == "blog":
        classifier = pipeline('text-classification', model="hallisky/blog-classifier-roberta-large", tokenizer="roberta-large", top_k=None, function_to_apply='sigmoid', device=device, truncation = True, model_kwargs = {"cache_dir":cache_dir})
    else:
        print("ERROR: No classifier for this author")  
        return(None)

    return(classifier)

# Pre-process data
def preprocess_data(text_ls: str) -> str:
   return [text.lower().translate(str.maketrans("", "", string.punctuation)).strip() for text in text_ls]

# Percentage of Function Words (using static list)
def eval_function_words_static(text_ls):
    # load list of function words
    functionWords = open("/src/writeprintresources/functionWord.txt", "r").readlines()
    functionWords = [f.strip("\\").replace("\n","") for f in functionWords]
    frequencyOfFunctionWords_ls = []
    number_of_words_ls = []
    # cycle through all texts
    for input_text in text_ls:
        # split text into words
        words = input_text.split()
        # cycle through all function words
        frequencyOfFunctionWords = []
        for i in range(len(functionWords)):
            functionWord = functionWords[i]
            freq = 0
            for word in words:
                if word == functionWord:
                    freq+=1
            frequencyOfFunctionWords.append(freq)
        frequencyOfFunctionWords_ls.append(np.sum(frequencyOfFunctionWords)) # number of function words
        number_of_words_ls.append(len(words)) # number of words
    return frequencyOfFunctionWords_ls, [f/w if w != 0 else 0 for f, w in zip(frequencyOfFunctionWords_ls, number_of_words_ls)]

# Percentage of Function Words (using parts of speech)
def eval_function_words(text_ls):
    n_function_words = []
    n_words = []
    for text in text_ls:
        translator = str.maketrans('', '', string.punctuation.replace("'", ""))
        words = text.translate(translator).split()
        tagged_words = pos_tag(words)
        # use parts of speech to identify "function" words
        function_words = [word for word, tag in tagged_words if tag not in ['NNP', 'NN' 'ADJ', 'ADV', 'FW', 'N', 'NP', 'NUM', 'VB', 'VBD', 'VBG', 'VBN', 'RB']] 
        n_function_words.append(len(function_words))
        n_words.append(len(text.split()))
    return n_function_words, [f/w if w != 0 else 0 for f, w in zip(n_function_words, n_words)]

# Grade Level
def eval_grade_level(text_ls):
    fk = []
    lw = []
    gf = []
    tx = []
    average = []
    for text in text_ls:
        fk_text = textstat.flesch_kincaid_grade(text)
        lw_text = textstat.linsear_write_formula(text)
        gf_text = textstat.gunning_fog(text)
        fk.append(fk_text)
        lw.append(lw_text)
        gf.append(gf_text)
        tx.append(textstat.text_standard(text))
        average.append(np.mean([fk_text, lw_text, gf_text]))
    return {"fk":fk, "lw":lw, "gf": gf, "tx":tx}, average

# Length
def eval_length(text_ls):
    num_words = []
    num_characters = []
    words_per_sent_ls = []
    char_per_sent_ls = []
    for text in text_ls:
        num_words.append(len(text.split()))
        num_characters.append(len(text))
        words_per_sent = []
        char_per_sent = []
        for sent in sent_tokenize(text):
            words_per_sent.append(len(sent.split()))
            char_per_sent.append(len(sent))
        words_per_sent_ls.append(np.mean(words_per_sent))
        char_per_sent_ls.append(np.mean(char_per_sent_ls))
    return num_words, num_characters, [c/w if w != 0 else 0 for c, w in zip(num_characters, num_words)], words_per_sent_ls, char_per_sent_ls
 
# Sarcasm
def eval_sarcasm(text_ls, device, model, tokenizer, batch_size = 8):
    confidence = []
    prediction = []
    batches = [text_ls[i:i+batch_size] for i in range(0, len(text_ls), batch_size)]
    for batch in batches:
        tokenized_text = tokenizer(preprocess_data(batch), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        output = model(**tokenized_text)
        _, pred = torch.max(output.logits.softmax(dim=1), dim = 1) # 1=sarcastic
        conf = output.logits.softmax(dim=1)[:,1]# confidence of being sarcastic
        confidence.extend(conf.tolist())
        prediction.extend(pred.tolist())
    return prediction, confidence

# Formality
def eval_formality(text_ls, device, model, tokenizer, batch_size = 8):
    confidence = []
    prediction = []
    batches = [text_ls[i:i+batch_size] for i in range(0, len(text_ls), batch_size)]
    for batch in batches:
        tokenized_text = tokenizer(preprocess_data(batch), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        output = model(**tokenized_text)
        _, pred = torch.max(output.logits.softmax(dim=1), dim = 1) # 1=formal
        conf = output.logits.softmax(dim=1)[:,1] # confidence of being formal
        confidence.extend(conf.tolist())
        prediction.extend(pred.tolist()) # 1 = formal
    return prediction, confidence

# Voice
def eval_voice(text_ls, device, model, tokenizer, batch_size = 8):
    confidence = []
    prediction = []
    batches = [text_ls[i:i+batch_size] for i in range(0, len(text_ls), batch_size)]
    for batch in batches:
        tokenized_text = tokenizer(preprocess_data(batch), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        output = model(**tokenized_text)
        _, pred = torch.max(output.logits.softmax(dim=1), dim = 1) # 1=active
        conf = output.logits.softmax(dim=1)[:,1] # confidence of being active
        confidence.extend(conf.tolist())
        prediction.extend(pred.tolist())
    return prediction, confidence

# Type
def calculate_percentages_and_average_confidences(prediction, confidence):
    total_count = len(prediction)
    percentages = [0] * 4
    confidence_sums = [0] * 4
    confidence_counts = [0] * 4
    
    for i in range(total_count):
        pred = prediction[i]
        conf = confidence[i]
        percentages[pred] += 1
        confidence_sums[pred] += conf
        confidence_counts[pred] += 1

    # Calculate percentages
    percentages = [(count / total_count) for count in percentages]

    # Calculate average confidences
    average_confidences = [
        (confidence_sums[i] / confidence_counts[i]) if confidence_counts[i] > 0 else 0
        for i in range(4)]

    return percentages, average_confidences

def eval_type(text_ls, device, model, tokenizer, batch_size = 8):
    confidence = []
    prediction = []
    all_confidence = []
    batches = [text_ls[i:i+batch_size] for i in range(0, len(text_ls), batch_size)]
    for batch in batches:
        tokenized_text = tokenizer(preprocess_data(batch), padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        output = model(**tokenized_text)
        conf, pred = torch.max(output.logits.softmax(dim=1), dim = 1) # 0 = Persuasive , 1 = Narrative, 2 = Expository, 3 = Descriptive
        confidence.extend(conf.tolist())
        prediction.extend(pred.tolist())
        all_confidence.append(output.logits.softmax(dim=1).tolist())
    percentages, average_confidences = calculate_percentages_and_average_confidences(prediction, confidence)
    return percentages, average_confidences, all_confidence, prediction

# NLI/Similarity 
# Check embedding similarity
sim_model = None
#Compute embedding for both lists
def compute_sim(original, rewrites, sim_model):
    if not isinstance(original, list):
        original = [original]
    if not isinstance(rewrites, list):
        rewrites = [rewrites]
    assert len(original) == len(rewrites), "inputs are different lengths"

    outputs = []
    embedding_orig= sim_model.encode(original, convert_to_tensor=True, show_progress_bar=False)
    embedding_rew = sim_model.encode(rewrites, convert_to_tensor=True, show_progress_bar=False)    
    
    outputs = cos(embedding_orig, embedding_rew).tolist()
    return outputs      

# Grammar 
def cola_score(text, cola_tokenizer, cola_model, device):
    tokenize_input = cola_tokenizer.tokenize(text)
    tensor_input = torch.tensor([cola_tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device)
    output=cola_model(tensor_input)
    return output.logits.softmax(-1)[0][1].item()

def eval_grammar(text_ls, cola_tokenizer, cola_model, device):
    cola_ls = []
    for i, text in enumerate(text_ls):
        if text == '':
            continue
        # if text is too big, break it up 
        if len(text) > 300:
            o = text
            print("The following original text is too big and will be broken up:", i)
            num_groups = int(np.ceil(len(o)/200))
            cola_score_list = []
            for n in range(num_groups):
                # cut it in half and add average to list
                o_split = o[n * int(len(o)/num_groups): (n+1) * int(len(o)/num_groups)]
                cola_score_list.append(cola_score(o_split, cola_tokenizer, cola_model, device))
            cola_ls.append(np.mean(cola_score_list))
        else:
            cola_ls.append(cola_score(text, cola_tokenizer, cola_model, device))
    return(cola_ls)

# Authorship Obfuscation
def eval_obfuscation(text_ls, author, obf_model):
    if "amt" in author:
        author = author.replace("amt_", "")
    if "blog" in author:
        index = int(author.replace("blog_5_", ""))
        static_blog_ls = ["5546","11518","25872","30102","30407"]
        author = "blog"+str(static_blog_ls[index-1])
    prediction_para = []
    confidence_para = []
    prediction_sent = []
    confidence_sent = []
    obf_para = []
    obf_sent = []
    # Initialize a dictionary to store the results
    for i, text in enumerate(text_ls):
        if text == "": 
            continue
        # PREDICTION BY PARAGRAPH  
        outputs = obf_model(text)[0]
        scores = []
        labels = []
        # extract scores for each label
        for item in outputs:
            label = item['label']
            score = item['score']
            labels.append(label)
            scores.append(score)
        prediction_para.append(labels[scores.index(np.max(scores))])
        if prediction_para[-1] == author: 
            obf_para.append(0)
        else:
            obf_para.append(1) # if incorrect, then do consider good (obfuscation rate)
        confidence_para.append(scores)
        
        # PREDICTION BY SENTENCE
        # create dictionary
        sent_prediciton_scores = {}
        for label_dict in outputs:
            sent_prediciton_scores[label_dict['label']] = []
            scores = []
            labels = []
        for sentence in sent_tokenize(text): 
            outputs = obf_model(sentence)[0]
            # extract scores for each label
            for item in outputs:
                label = item['label']
                score = item['score']
                labels.append(label)
                sent_prediciton_scores[label].append(score)
        sent_averages = [np.mean(sent_prediciton_scores[key]) for key in sent_prediciton_scores.keys()]
        try:
            prediction_sent.append(labels[sent_averages.index(np.nanmax(sent_averages))])
        except:
            continue
        if prediction_sent[-1] == author:
            obf_sent.append(0)
        else:
            obf_sent.append(1) # if incorrect, then do consider good (obfuscation rate)
        confidence_sent.append(sent_averages)
    return(prediction_para, confidence_para, obf_para, prediction_sent, confidence_sent, obf_sent)


        
def automatic_eval(dataset, device, obfuscation = False, author = None, content_overlap = False, original = None, sarcasm_model= None, sarcasm_tokenizer= None, voice_model= None, voice_tokenizer= None, type_model= None, type_tokenizer= None, formality_model= None, formality_tokenizer= None, cola_model= None, cola_tokenizer= None, classifier= None, sim_model= None):
    dataset = [unidecode(d) for d in dataset]
    n_words, n_characters, char_per_word, words_per_sent, char_per_sent = eval_length(dataset)
    n_function_words, percent_function_words = eval_function_words(dataset)
    grade_level_dict, grade_level_average = eval_grade_level(dataset)
    formal_pred, formal_conf = eval_formality(dataset, device, formality_model, formality_tokenizer)
    sarcasm_pred, sarcasm_conf = eval_sarcasm(dataset, device, sarcasm_model, sarcasm_tokenizer)
    voice_pred, voice_conf = eval_voice(dataset, device, voice_model, voice_tokenizer, batch_size = 8)
    type_pred, type_conf, type_overall_confidence, type_overall_prediction = eval_type(dataset, device, type_model, type_tokenizer, batch_size = 8)
    cola_ls = eval_grammar(dataset, cola_tokenizer, cola_model, device)

    results = {"n_words": n_words,
                "n_characters": n_characters,
                "char_per_word": char_per_word,
                "words_per_sent": words_per_sent,
                "char_per_sent": char_per_sent,
                "n_function_words": n_function_words,
                "percent_function_words": percent_function_words,
                "grade_level_dict": grade_level_dict,
                "grade_level_average": grade_level_average,
                "formal_pred": formal_pred,
                "formal_conf": formal_conf,
                "sarcasm_pred": sarcasm_pred,
                "sarcasm_conf": sarcasm_conf,
                "voice_conf": voice_conf,
                "voice_pred": voice_pred,
                "type_persuasive_conf": type_conf[0],
                "type_persuasive_pred": type_pred[0], 
                "type_narrative_conf": type_conf[1],
                "type_narrative_pred": type_pred[1], 
                "type_expository_conf": type_conf[2],
                "type_expository_pred": type_pred[2], 
                "type_descriptive_conf": type_conf[3],
                "type_descriptive_pred": type_pred[3], 
                "type_overall_conf": type_overall_confidence,
                "type_overall_pred": type_overall_prediction,
                "grammar": cola_ls}
    
    if obfuscation:
        # Using Roberta based Classifiers
        obf_pred_para, obf_conf_para, obf_para, obf_pred_sent, obf_conf_sent, obf_sent = eval_obfuscation(dataset, author, classifier[0])
        results["obf_pred_para_Roberta"] = obf_pred_para
        results['obf_conf_para_Roberta'] = obf_conf_para
        results['obf_rate_para_Roberta'] = obf_para
        results["obf_pred_sent_Roberta"] = obf_pred_sent
        results['obf_rate_sent_Roberta'] = obf_sent
        results['obf_conf_sent_Roberta'] = obf_conf_sent
            
    if content_overlap:
        similarity = compute_sim(original, dataset, sim_model)
        results["similarity"] = similarity
    return(results)

    