# Political-Affiliation-with-BERT

Uploaded for safekeeping.  The code here was used to train a BERT model to classify politicians by party affiliation.  The data we used to train our model is included in test_party_IDs.csv.  The logistic model was included as a record, but classifytweets.py, which uses a linear neural network classifier, is what we found the most success with.  We achieve around 86% accuracy with this method. 

The majority of the project is contained in three files: "data_collection_&_processing.ipynb", "classifytweets.py", and "evaluation.ipynb". "data_collection_&_processing.ipynb" is a Jupyter notebook that contains all of the steps required to generate the data we used to train the model.

"data_collection_&_processing.ipynb" steps:
    - Drop 3rd party politicians.  
    - Drop politicians unactive on twitter.
    - Use twitter API to get 1000 tweets (not retweets) for each user.  Write tweets into csvs corresponding to each user, one tweet per row.
    - Filter out users with less than 1000 tweets.
    - Combine all tweets from 1000 tweet users into one csv.
    - Assign labels.

"classifytweets.py" is where all of the modeling is done.  There is also some data parsing.

"classifytweets.py" steps:
    - Balance the data (we have more Dems than Reps after the list is filtered down).
    - Assign each User IDs into either the training, validation or test set (we used a 50/30/20 split).
    - Concatonate tweets belonging to each politician into an odd number of <500 word chunks (BERT can only handle 512 tokens at a time).  Use longest tweets to construct the chunks.
    - Construct training/validation/test set out of the chunks belonging to each politician assigned to the training/validation/test set.
    - Use the tweet chunks to train BERT paired with a linear NN classifier.  This tends to take 7-10 hours.
    - Once BERT has trained over however many EPOCHS specified (we had the most success with 5), a csv containg the results for the run will be written.



Plans for the future:

- Review dataset.  We have found a few samples where party affiliation is unfortunately mislabeled.
- Add parallel computing functionallity.  This code takes around 10 hours to run with ~100k tweets.  We want more than that, but without threading and/or better hardware running on more tweets is impractical.
- Long(er) term: Add a second layer of classification to sort out political tweets from non-political ones.  Non-political tweets introduce a lot of noise and and make some samples difficult to classify.  Would be a big step towards expanding the system to work on general-population users.
