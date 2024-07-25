# Training Author Style Classifiers

This folder contains the necessary code to train classifiers for various author styles. For instance, we can train a discriminator to determine whether a given text is written by Trump, Obama, or Bush (the speech category).

`utils.py` contains helper functions, while `train.py` contains the code required to train the model. Note that the dataset needed to train classifiers, AuthorMix, is stored under https://huggingface.co/datasets/hallisky/AuthorMix/.

## Training the Model

We can run the following code to train a classifier on speeches (obama, bush, trump). 
```
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
```

By default, this uses RoBERTa-large. Feel free to look at the `argparser` and modify other training arguments.

To train classifiers on the other style categories in AuthorMix, we need to substitute out what we pass into `--styles`. Specifically, we use the following for each of the categories:
* Author: `--styles woolf fitzgerald hemingway`
* Blog: `--styles blog5546 blog11518 blog25872 blog30102 blog30407`
* AMT: `--styles h pp qq`
* Speech (shown above): `--styles obama bush trump`

## Evaluating the Model 
To evaluate the trained classifier, we simply pass the `--evaluate` flag and pass in the `--pretrained_path` to a save checkpoint. 

Here is an example evaluation command on the speech category (Obama, Bush, Trump):

```
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
```

This will generate the per-class and total accuracies of the pretrained model on the evaluation dataset of the `hf_data_path` dataset.