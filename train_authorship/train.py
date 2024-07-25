"""
Trains a classifier on the specified Author styles. By default,
uses RoBERTa-large. See argparser for more hyperparameters.

# Example to train a classifier on speeches (obama, bush, trump)
CUDA_VISIBLE_DEVICES=0 python3 -m train_authorship.train \
    --use_accuracy_for_training \
    --lr 5e-5 \
    --batch_size 64 \
    --seed 0 \
    --epochs 10 \
    --save_ratio 2 \
    --styles obama bush trump \
    --max_length_tok 256 \
    --hf_data_path hallisky/AuthorMix

# Example to evaluate the trained classifier on speeches (obama, bush, trump)
CUDA_VISIBLE_DEVICES=0 python3 -m train_authorship.train \
    --use_accuracy_for_training \
    --lr 5e-5 \
    --batch_size 64 \
    --seed 0 \
    --epochs 10 \
    --save_ratio 2 \
    --styles obama bush trump 
    --max_length_tok 256 \
    --hf_data_path hallisky/AuthorMix \
    --evaluate \
    --pretrained_path [model_path_to_evaluate]
    
# To train on authors, use the following --styles
    --styles woolf fitzgerald hemingway \

# To train on blog, use the following --styles
    --styles blog5546 blog11518 blog25872 blog30102 blog30407 \

# To train on AMT, use the following --styles
    --styles h pp qq \
"""

import os
os.environ['TRANSFORMERS_CACHE'] = '../cache/'
os.environ["WANDB_DISABLED"] = "true" 

import argparse
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    set_seed
)

import random
import math
from train_authorship.utils import read_from_jsonl
import numpy as np
from utils import clean_text
from sklearn.utils import resample
from datetime import datetime
time = datetime.now()    
date_time = time.strftime("%m-%d-%Y_%H:%M:%S")
from datasets import load_dataset

def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    num_labels = len(args.styles)
    print(f"We have {num_labels} styles")

    if not args.evaluate: # Train model from scratch
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=num_labels)
    else: # Load existing model for evaluation only
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_data, train_labels = [], []
    dev_data, dev_labels = [], []

    # Make a dict to map style to number and vice versa
    style_dict = {label: i for i, label in enumerate(args.styles)}
    rev_style_dict = {i: label for i, label in enumerate(args.styles)}
    # Set label2id and id2label configurations
    model.config.label2id = style_dict
    model.config.id2label = rev_style_dict

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Make sure user passes in a hf dataset path; if they don't, prompt them to implement their own logic
    if args.hf_data_path is None:
        print("Exiting since no huggingface data path was passed. Feel free to implement your own data loading\
              logic with local paths or other data.")
        import sys; sys.exit()
    
    # Loading in and processing the data
    data = load_dataset(args.hf_data_path)
    final_data = {}

    # Balance train data (upsample or downsample); leave dev data imbalanced.
    for split in ["train", "validation"]:
        split_data = data[split]
        cur_data = {}
        for cur_style in args.styles:
            cur_data[cur_style] = split_data.filter(lambda x: x["style"] == cur_style)["text"]
        
        comb_data, comb_labels = [], []
        if split == "train": # Train dataset
            if args.upsample: # Upsample data to maximum length
                max_length = 0
                for s in cur_data.keys():
                    max_length = max(len(cur_data[s]), max_length)
                print(f"Max length for split {split} is {max_length}")

                for s in cur_data.keys():
                    comb_data.extend(resample(cur_data[s], n_samples=max_length))
                comb_labels.extend([style_dict[s]] * max_length)

            else: # Downsample data to minimum length
                min_length = 1e6 # Arbitrarily large value
                for s in cur_data.keys():
                    min_length = min(len(cur_data[s]), min_length)
                print(f"Min length for split {split} is {min_length}")

                for s in cur_data.keys():
                    comb_data.extend(random.sample(cur_data[s], min_length))
                    comb_labels.extend([style_dict[s]] * min_length)

        else: # Evaluation dataset
            for s in cur_data.keys():
                comb_data.extend(cur_data[s])
                comb_labels.extend([style_dict[s]] * len(cur_data[s]))
                print(s, len(cur_data[s]))
        final_data[split] = (comb_data, comb_labels)

    train_data, train_labels = final_data["train"]
    dev_data, dev_labels = final_data["validation"]
    print(f"Length of train data is {len(train_data)}, length of val data is {len(dev_data)}")

    # Shuffle the data
    train_combo = list(zip(train_data, train_labels))
    random.shuffle(train_combo)
    train_data, train_labels = zip(*train_combo)
    train_data, train_labels = list(train_data), list(train_labels)

    dev_combo = list(zip(dev_data, dev_labels))
    random.shuffle(dev_combo)
    dev_data, dev_labels = zip(*dev_combo)
    dev_data, dev_labels = list(dev_data), list(dev_labels)
    
    # Print train sequence length statistics
    train_lengths = np.array([len(a) for a in tokenizer(train_data)["input_ids"]])
    print("99th percentile, median, max are:\n",np.percentile(train_lengths, 99), np.median(train_lengths), np.max(train_lengths))

    # Collate function for batching tokenized texts
    def collate_tokenize(data, max_length_tok=args.max_length_tok):
        text_batch = [element["text"] for element in data]
        tokenized = tokenizer(text_batch, padding='longest', truncation=True, return_tensors='pt', max_length=max_length_tok)
        label_batch = [element["label"] for element in data]
        tokenized['labels'] = torch.tensor(label_batch)
        return tokenized

    class StyleDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.texts = texts
            self.labels = labels

        def __getitem__(self, idx):
            item = {}
            item['text'] = self.texts[idx]
            item['label'] = self.labels[idx]
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = StyleDataset(train_data, train_labels)
    dev_dataset = StyleDataset(dev_data, dev_labels)

    # Number of steps per epoch
    steps = len(train_labels)/(args.batch_size*torch.cuda.device_count())
    save_steps = math.ceil(steps / args.save_ratio) # Save every 1/args.save_ratio
    print(f"Save steps is {save_steps}")

    # Training branch
    if not args.evaluate:
        compute_metrics= None
        metric_for_best_model = None
        greater_is_better = None

        # If we want to calculate classification accuracy while we're training
        if args.use_accuracy_for_training:
            def accuracy(eval_pred):
                predictions, labels = eval_pred
                predictions = torch.argmax(torch.tensor(predictions), dim=-1).tolist()
                
                # Initialize dictionaries to track correct and total predictions per class
                correct_predictions_per_class = {}
                total_predictions_per_class = {}
                
                # Loop through all predictions and labels
                for pred, label in zip(predictions, labels):
                    if label not in total_predictions_per_class:
                        total_predictions_per_class[label] = 0
                        correct_predictions_per_class[label] = 0
                    total_predictions_per_class[label] += 1
                    
                    if pred == label:
                        correct_predictions_per_class[label] += 1
                
                # Calculate accuracy for each class
                accuracy_per_class = {rev_style_dict[label] + "_acc": correct / total for label, correct, total in zip(total_predictions_per_class.keys(), correct_predictions_per_class.values(), total_predictions_per_class.values())}
                
                # Calculate overall accuracy
                overall_accuracy = sum(correct_predictions_per_class.values()) / sum(total_predictions_per_class.values())
                assert overall_accuracy == sum([a == b for a, b in zip(predictions, labels)])/len(predictions)

                # Combine overall accuracy with per-class accuracies
                accuracy_results = {'overall_acc': overall_accuracy, 'acc_product': np.prod(list(accuracy_per_class.values())), **accuracy_per_class}
                return accuracy_results
            
            compute_metrics = accuracy
            metric_for_best_model = 'overall_acc'
            greater_is_better = True

        args = TrainingArguments(
            output_dir = os.path.join(args.output_dir,date_time), 
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            evaluation_strategy='steps',
            num_train_epochs=args.epochs,
            eval_steps = save_steps,
            save_steps = save_steps,
            logging_steps = save_steps,
            lr_scheduler_type = 'linear',
            learning_rate=args.lr,
            seed = args.seed,
            warmup_ratio = 0.1,
            load_best_model_at_end = True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            remove_unused_columns=False
            )

        trainer = Trainer(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=dev_dataset, 
            tokenizer=tokenizer,
            data_collator = collate_tokenize,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(10)] # Early stopping callback of 10 evaluations
            )

        trainer.train()

    else: # Evaluation logic
        model.eval()
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        dataloader = DataLoader(dev_dataset, collate_fn=collate_tokenize, batch_size=args.batch_size)

        truth, pred = [], []
        class_predictions = {}  # Dictionary to store prediction counts for each true class

        # Iterating through each batch in the DataLoader
        for d in tqdm(dataloader):
            true_labels = [d.item() for d in d["labels"]]
            predictions = torch.argmax(model(**d).logits, dim=-1).tolist()

            truth.extend(true_labels)
            pred.extend(predictions)

            # Update prediction counts for each true class
            for true, predicted in zip(true_labels, predictions):
                true_label = model.config.id2label[true]
                pred_label = model.config.id2label[predicted]

                if true_label not in class_predictions:
                    class_predictions[true_label] = {}
                if pred_label not in class_predictions[true_label]:
                    class_predictions[true_label][pred_label] = 0
                class_predictions[true_label][pred_label] += 1

        # Calculate and print accuracy
        accuracy = sum([a == b for a, b in zip(truth, pred)]) / len(pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Print prediction counts for each true class
        print("Prediction counts for each true class:")
        for true_class, predictions_dict in class_predictions.items():
            print(f"True Class: {true_class}, Predictions: {predictions_dict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate a style classification model")
    parser.add_argument(
        '--model', type=str, default="roberta-large",
        help="Model architecture (Huggingface path) to train"
    )
    parser.add_argument(
        '--hf_data_path', type=str, default=None,
        help="Path to the Huggingface dataset containing the training and validation data. If not passed in, script will prompt user to implement their own data loading logic."
    )
    parser.add_argument(
        '--styles', type=str, nargs="+", default=["amt-3-h", "amt-3-pp", "amt-3-qq", "heming", "obama", "trump"],
        help="List of styles to be used for training the model"
    )
    parser.add_argument(
        '--save_ratio', type=int, default=4,
        help="Frequency of saving the model during training (e.g., save every 1/save_ratio steps)"
    )
    parser.add_argument(
        '--lr', type=float, default=5e-5,
        help="Learning rate for the optimizer"
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help="Number of epochs to train the model"
    )
    parser.add_argument(
        '--max_length_tok', type=int, default=128,
        help="Maximum length of tokenized sequences"
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        '--seed', type=int, default=1,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output_dir', type=str, default='train_authorship/train_outputs',
        help="Directory to save the trained model and outputs"
    )
    parser.add_argument(
        '--pretrained_path', type=str, default=None,
        help="Path to a pretrained model for evaluation"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="If set, the script will run in evaluation mode"
    )
    parser.add_argument(
        "--upsample", action="store_true",
        help="If set, the training data will be upsampled to the size of the largest class"
    )
    parser.add_argument(
        "--use_accuracy_for_training", action="store_true",
        help="If set, use accuracy as the evaluation metric during training"
    )
    main(parser.parse_args())
