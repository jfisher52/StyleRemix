# StyleRemix: An Intepretable Authorship Obfuscation Method
This repository contains the code and the scripts to reproduce the experiments from the paper
[StyleRemix: Interpretable Authorship Obfuscation via Distillation and Perturbation of Style Elements](). 

**StyleRemix**, is an adaptive and interpretable obfuscation method that perturbs specific, fine-grained style elements of the original input text. StyleRemix uses pre-trained Low Rank Adaptation (LoRA) modules to rewrite inputs along various stylistic axes (e.g., formality, length) while maintaining low computational costs. 

<p align="center">
<img src="styleremix_overview.jpg" width="275">
</p>

In this paper we demonstrate the effectiveness of StyleRemix on four obfuscation datasets comprised of presidential speeches (SPEECHES), fiction writing (NOVELS), academic articles (SCHOLAR) and diary-style writings (BLOG). When applied in combination with a LLAMA-3 7B model, StyleRemix outperforms state-of-the-art authorship baselines and much larger LLMs on an array of domains on both automatic and human evaluation.

In this repo, we provide code which implements StyleRemix on a LLAMA-3 8B model for these four datasets.

## Using this Repository
All code is meant to be run in Python.


### Setting up the Environment
To set up the environment to run the code, make sure to have conda installed, then run

    conda create --name obf python=3.10

Then, activate the environment

    conda activate obf

Finally, install the required packages (make sure you are in the root directory).

    pip install -r requirements.txt

## Resources

All resources (trained models, demo, etc.) have been organized into the following [huggingface collection](https://huggingface.co/collections/hallisky/authorship-obfuscation-66564c1c1d59bb62eaaf954f). 
    
## Datasets
We use the AuthorMix data which is composed offour different domains, presidential speeches (curated in this paper), fictional novels (curated in this paper), the Extended-Brennan-Greenstadt (Brennan et al., 2012) (amt) and the  Blog Authorship corpus (Schler et al., 2006) (blog), using a range of different authors (3 - 5). All raw datasets can be found under the  `test_data/` folder. Note the file `test_data/AuthorMix` is a torch file with a dictionary containing a key for each domain (Speeches, Novels, AMT, Blog) and the file `test_data/AuthorMix_average_by_author` contains a pre-computed matrix of average automatic evalution by author which is used to choose the weights of the adapters. 

The test dataset can also be downloaded directly from huggingface: [link](https://huggingface.co/datasets/hallisky/AuthorMix)

```
from datasets import load_dataset
data = load_dataset("hallisky/AuthorMix")
```


## Experimental Pipeline
Experimental code for both all domains can be found in the main folder labeled as `obfuscate.py`. Each experiment consists of the following four steps:

1. Download Raw Data:  automatic
2. Run Automatic Evaluation: automatic if using current domains
3. Choosing Styles to Perturb for Adapters: less than 1 minute
4. Choose Weight of Adapters: varies by AuthorMix method used (lorahub ~10min, sequential <1min)
5. Obfuscation: varies by AuthorMix method used (lorahub ~5min, sequential 5min/style)
6. Evaluation: ~10min


## Citation
If you find this repository useful, or you use it in your research, please cite:
```

```
    
## Acknowledgements

## Contact

If you have any issues with the repository, questions about the paper, or anything else, please email jrfish@uw.edu and hallisky@uw.edu.

