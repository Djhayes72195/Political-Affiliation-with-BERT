# Political-Affiliation-with-BERT

Uploaded for safekeeping.  The code here was used to train a BERT model to classify politicians by party affiliation.  More data could be generated with the data processing notebook here, but you would need your own bearer token.  The logistic model was included as a record, but classifytweets.py, which uses a linear neural network classifier, is what we found the most success with.  We achieve around 86% accuracy with this method. 

The pdf "Using BERT to Predict the Political Ideology of Twitter Users" contains a full writeup, but it is pretty long.  Here are the main points:

- We used the Twitter API to attempt to fetch 1000 tweets from ~800 politicians.  A few IDs failed to retrieve all 1000 tweets from the API; we have excluded those politicians from the analysis for now.  The IDs of each politician were obtained from a Kaggle dataset which we cleaned up a bit (filtered for twitter activity, removed duplicate accounts, removed independent politicians, etc). The post-cleaning Kaggle dataset is called "cleanest_politicians.csv" here. Most of the code needed to do all of this is included in the Jupyter notebook "data_collection&processing.ipynb".  

- The code used to fine-tune the BERT embeddings and classify by party is included in "classifytweets.py". We used a linear  neural network for classification.  Training/test/validation split is also handled in classifytweets.py.  After the       split, a the subset of our data that we use for training and classification is automatically sampled and processed as follows: 
  - An odd number of tweet chunks to represent each politician is specified.  Each tweet chunk is a concatentation of           tweets by a particular politician.  The code functions so as to not allow any tweet chunk to contain more than 500         words, as BERT standard can only consider a maximum of 512 characters at a time.

Plans for the future:

- Review dataset.  We have found a few samples where party affiliation is unfortunately mislabeled.
- Add parallel computing functionallity.  This code takes around 10 hours to run with ~100k tweets.  We want more than that, but without threading and/or better hardware running on more tweets is impractical.
- Long(er) term: Add a second layer of classification to sort out political tweets from non-political ones.  Non-political tweets introduce a lot of  noise and and make some samples difficult to classify.  Would be a big step towards expanding the system to work on general-population users.
