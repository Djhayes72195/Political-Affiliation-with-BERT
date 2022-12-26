### BERT Model and Logistic Regression Classifier
### This is an example of BERT and Logistic Regression
### This script was not run properly due to the large amount of data size
### The data that was used was the full list of observed individual tweets regardless of political parties and politicians

# Required packages
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')
import pickle

# change working directory
os.chdir('C:/Users/Francis Jo/Desktop/STAT766_temp/03data/politician_csvs') # laptop
politician=os.listdir() # gives the list of all names of files in a current working directory.
# concatenate all csv files
polittwitterall = pd.DataFrame() # Empty Dataframe
for polit in politician:
    polit_twitter = pd.read_csv(polit,engine='python')
    polittwitterall = pd.concat([polittwitterall,polit_twitter],ignore_index=True) # keys : Unnamed: 0, TweetID, UserID, Text
# check any NA/nan/None values for each row
n_total = polittwitterall.shape[0]
for i in range(0,n_total):
    if np.isnan(polittwitterall['UserID'][i]):
        print(i) # there's an issue with some observations # 1506, 130815, 131077, 297848, 475564, 784065
        print(polittwitterall.iloc[i])
        polittwitterall['Text'][i-1] = str(polittwitterall['Text'][i-1]) + ' ' + str(polittwitterall['Unnamed: 0'][i])
# remove/drop all the rows with Na/nan/None values
polittwitterall.dropna(inplace=True)
polittwitterall.reset_index(drop=True, inplace=True)
print(polittwitterall) # print the cleaned full data


# list of political party for each row in polittwitterall dataframe
os.chdir('C:/Users/Francis Jo/Desktop/STAT766_temp/03data') # laptop
#os.chdir('C:/Users/fsjo1/Desktop/Active/Fall2022/STAT766/s766_03_data') # desktop
# call the list of politicians and clean the categories of political party
polit_party = pd.read_csv('cleanest_politicians.csv') # load the csv file with the list of political parties
n_polit_party = polit_party.shape[0]
for i in range(0,n_polit_party):
    currentrow = polit_party['Political_party'][i]
    if currentrow != 'Democratic Party' or currentrow != 'Republican Party':
        if 'Democratic' in currentrow:
            polit_party['Political_party'][i] = 'Democratic Party'
        if 'Republican' in currentrow:
            polit_party['Political_party'][i] = 'Republican Party'

# Dataset option 1: Dataset with a list of 885 politicians and all merged tweet text for each politician
# merge all the texts/tweets into one full string for each politician
n_total = polittwitterall.shape[0]
text_list = np.empty([n_polit_party,1],dtype='object') # empty list to fill in the text for each politician
for i in range(0,n_total):
    currentindex = np.where(polit_party['Account_ID']==polittwitterall['UserID'][i])[0][0]
    if text_list[currentindex,0] == None:
        text_list[currentindex,0] = polittwitterall['Text'][i]
    if text_list[currentindex,0] != None:
        text_list[currentindex,0] = text_list[currentindex,0] + ' ' + polittwitterall['Text'][i]
# bind together with polit_party dataframe
text_list = pd.DataFrame(text_list,columns=['Text']) # convert text_list into pandas dataframe
for i in range(0,len(text_list)):
    if text_list['Text'][i] == None:
        print(i)
polit_party = pd.concat([polit_party,text_list], axis=1) # bind them together
# remove/drop all the rows with Na/nan/None values
polit_party.dropna(inplace=True)
polit_party.reset_index(drop=True, inplace=True)

# Data option 2: list of all individual tweet texts, where each tweet text has the type of political party (0 and 1 code)
# obtain the political party list
n_total = polittwitterall.shape[0]
politparty_list = np.empty([n_total,1],dtype='object') # empty list to fill in the political party for each row
for i in range(0,n_total):
    currentindex = np.where(polit_party['Account_ID']==polittwitterall['UserID'][i])[0][0]
    currentpolitparty = polit_party['Political_party'][currentindex]
    if currentpolitparty == 'Democratic Party' or currentpolitparty == 'Republican Party':
        politparty_list[i,0] = currentpolitparty
    if 'Democratic' in currentpolitparty:
        politparty_list[i,0] = 'Democratic Party'
    if 'Republican' in currentpolitparty:
        politparty_list[i,0] = 'Republican Party'
#bind together with plittwitterall
politparty_list = pd.DataFrame(politparty_list,columns=['Political party']) # convert plitparty list into pandas dataframe
polittwitterall = pd.concat([polittwitterall,politparty_list], axis=1) # bind them together
print(polittwitterall)
polittwitterall_v2 = pd.concat([polittwitterall['Political party'],polittwitterall['Text']], axis=1)
# obtain the political party in terms of binary numbers
partycode = np.empty([n_total,1],dtype='int64') # empty list to fill in the political party with code; 1 = democratic, 0 = republican
for i in range(0,n_total):
    currentpolitparty = polittwitterall_v2['Political party'][i]
    if currentpolitparty == 'Democratic Party':
        partycode[i,0] = 1
    if currentpolitparty == 'Republican Party':
        partycode[i,0] = 0
partycode = pd.DataFrame(partycode,columns=['Code'])
polittwitterall_v3 = pd.concat([polittwitterall['Text'],partycode], axis=1)

polittwitterall_v3.to_csv('politician_alltweets.csv', header=True)

# change working directory
#os.chdir('D:/STAT766_PROJECT') # desktop
os.chdir('C:/Users/Francis Jo/Desktop/STAT766_temp/03data')

# call the dataset
df = pd.read_csv('politician_alltweets.csv')
batch_1 = df[:2000] # portion of the full data
print(batch_1['Code'].value_counts())

##################################
# Load the pre-trained BERT model
##################################
# distilBERT model - "a version of BERT that is smaller, but much faster and requiring a lot less memory"
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
# BERT model
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

##################################
# Preparing the Dataset (Tokenize the texts)
##################################
# tokenize the sentences - break them up into word and subwords in the format BERT is comfortable with.
tokenized = batch_1['Text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# PADDING - pad all lists to the same size to represent the inputs for BERT as one 2d-array
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i) # max length of 113 for the first 2000 observations
#max_len = 290 # the max length of the tokenized values among all observations in the full data
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
np.array(padded).shape # check the dimension of the padded sentences with same length size

# MASKING - create another variable to tell BERT to ignore (mask) the padding we've added when it's processing its input
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape # check the dimension of the masked sentences

##################################
# MODEL1: Deep Learning
##################################
# run the model using the tokenized, padded, and attention_mask
input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

# Get features from the tokenized objects
## WARNING: THIS IS WHERE WE FAILED TO GET FEATURES DUE TO THE LARGE AMOUNTS OF DATA.
## DEPENDING ON WHICH COMPUTERS CURRENTLY BEING USED AND HOW MUCH DATA YOU HAVE, IT MAY OR MAY NOT RUN PROPERLY.
# def findfeat(input_ids,attention_mask):
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
#     return last_hidden_states[0][:,0,:].detach().numpy()
# features = findfeat(input_ids=input_ids, attention_mask=attention_mask)

# Below is the embedding vectors obtained by inputing one tweet observation at a time
# (probably) not a correct way to get the embedding vectors
features = np.empty((0,768))
for i in range(batch_1.shape[0]):
    last_hidden_states = model(input_ids[i:(i+1),], attention_mask=attention_mask[i:(i+1),])
    last_hidden_states = last_hidden_states[0][:,0,:].detach().numpy()
    features = np.concatenate([features,last_hidden_states],axis=0)

labels = batch_1['Code'] # indicates which sentence is positive and negative; used for later.

##################################
# MODEL 2: Train/Test Split
##################################
# Split the dataset into a training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

# find the best (optimal) value of the C parameter, which determines regularization strength (i.e. Cross-Validation)
# This one is optional
parameters = {'C': np.linspace(0.0001, 100, 20)}
grid_search = GridSearchCV(LogisticRegression(), parameters)
grid_search.fit(train_features, train_labels)
print('best parameters: ', grid_search.best_params_)
print('best scrores: ', grid_search.best_score_)

# Train the Logistic Regression model
# if the grid search was done above, plug the found best value of C into the model declaration (e.g. LogisticRegression(C=5.2))
#lr_clf = LogisticRegression(C=5.2)
lr_clf = LogisticRegression(C = 42.10532105263158) # optimal tuning parameter: C = 42.105
lr_clf.fit(train_features, train_labels)
#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None, max_iter=100,
# multi_class='warn', n_jobs=None, penalty='l2', random_state=None, solver='warn', tol=0.0001, verbose=0,warm_start=False)

# SAVE THE LOGISTIC REGRESSION MODEL TO DISK
filename = 'finalized_model.sav'
pickle.dump(lr_clf, open(filename, 'wb'))

# LOAD THE LOGISTIC REGRESSION MODEL FROM DISK
loaded_model = pickle.load(open(filename, 'rb'))

# Evaluate the model; check the accuracy against the testing dataset
lr_clf.score(test_features, test_labels)
lr_clf.predict(test_features) # predict classes using the test_features