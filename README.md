# Political Affiliation Classification with BERT

This project uses a fine-tuned BERT model to classify Twitter users by political affiliation, based on their tweet history. The dataset includes ~750,000 tweets from U.S. politicians, and the model consistently achieves ~85% user-level accuracy. This project was initially developed with politicians for ground truth labeling, with the goal of generalizing to the broader Twitter user base. Additional development would be required to generalize this model.

**Disclaimer**: This project was developed in 2022. It was an early exploration of NLP and BERT-based classification. The code is functional, and the project was a valuable learning experience, but my more recent work demonstrates stronger engineering practices. See my [semantic search engine](https://github.com/Djhayes72195/SemanticSearch) for an example of my current work.

## Authors

This project was created by Dustin Hayes, Francis Jo, and Nicholas Mackay.

Core implementation, data processing, and model development were primarily done by Dustin Hayes.

## Key Features

- Fine-tuned BERT model on labeled tweet chunks (~750k tweets)
- Achieves ~85% user-level test accuracy
- Built with PyTorch, Pandas, and Jupyter


## Use Cases

- Classifying users to improve segmentation in social media research
- Enhancing sentiment analysis by segmenting by political affiliation
- Exploring latent political signals in social media datasets

**Ethical Note:** This project is for research purposes only. Profiling users on the basis of political ideology is a sensitive task and should not be done without careful consideration of ethical 
implications.
---

## Installation

1. Install [Git LFS](https://git-lfs.com/) and run:
```
git lfs install
```

2. Clone the repo and install dependencies.
```
git clone https://github.com/Djhayes72195/Political-Affiliation-with-BERT.git
cd Political-Affiliation-with-BERT
pip3 install -r requirements.txt
```
3. To train the model (slow without access to GPU):
```
python classifytweets.py
```

4. To interact with the notebooks:
```
pip3 install jupyterlab
jupyter-lab
```


## Project Structure
- data_collection_&_processing.ipynb: Collects tweets, cleans text, and outputs labeled CSV
- classifytweets.py: Prepares tweet chunks, trains BERT, and generates predictions
- evaluation.ipynb: Calculates user-level accuracy and displays basic metrics
