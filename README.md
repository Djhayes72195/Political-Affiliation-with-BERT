# Political-Affiliation-with-BERT

Uploaded for safekeeping.  The code here was used to train a BERT model to classify politicians by party affiliation.  More data could be generated with the data processing notebook here, but you would need your own bearer token.  The logistic model was included as a record, but classifytweets.py, which uses a linear neural network classifier, is what we found the most success with.  We achieve around 86% accuracy with this method. 

Plans for the future:

- Review dataset.  We have found a few samples where party affiliation is unfortunately mislabeled.
- Add parallel computing functionallity.  This code takes around 10 hours to run with ~100k tweets.  We want more than that, but without threading and/or better hardware running on more tweets is impractical.
- Long(er) term: Add a second layer of classification to sort out political tweets from non-political ones.  Non-political tweets introduce a lot of  noise and and make some samples difficult to classify.  Would be a big step towards expanding the system to work on general-population users.
