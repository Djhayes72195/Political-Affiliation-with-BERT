import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import tweepy
import numpy as np
from transformers import BertTokenizer
import torch
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from collections import Counter
import random
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_into_512_approx(users, tweet_chunks_per_user, all_info):
    """This function will take in a list of users and construct either test,
    validation, or training dataframe.  tweet_blocks per user specifies how many chunks of tweets
    we wish to include for each user.  Each tweet chunk is a concatenation of tweets by a particular user.
    This function will ensure that no tweet chunk contains more than 500 words, as BERT can only handle
    512 tokens at a time. The chunks will be constructed with the longest tweets first, on the
    assumption that longer tweets would be more likely to contain political messaging.

    We could have run BERT with individual tweets as opposed to tweet chunks, and we did try that
    many times.  We aren't exactly sure why, but chunking the tweets together seemed to produce
    better results and made computation faster.

    This function was constructed so as to ensure that tweets from any particular
    politician only appear in only the test, validation or training set, not some
    combination of the three.
    """
    df_out = pd.DataFrame(columns=['Party', 'Text', 'UserID'])
    num_dem = 0
    num_rep = 0
    counter2 = 0
    for user in users:
        temp_df = all_info.loc[all_info['UserID'] == user]
        if temp_df['Party'].iloc[0] == 'Republican Party':
            num_rep += 1
        if temp_df['Party'].iloc[0] == 'Democratic Party':
            num_dem += 1
        max_indexes = []
        temp_texts = list(temp_df['Text'])
        lengths = np.zeros(len(temp_texts))
        for i in range(0, len(temp_texts)):
            lengths[i] = len(temp_texts[i].split(' '))
        for i in range(0, len(temp_texts)):
            temp_max_indi = lengths.argmax()
            lengths[temp_max_indi] = 0
            max_indexes.append(temp_max_indi)
        counter = 0
        big_counter = 0
        working_string = ''
        for index in max_indexes:
            data = temp_df['Text'].iloc[index]
            if counter >= tweet_chunks_per_user:
                break
            if len((working_string + ' ' + data).split(' ')) > 500:
                df_out = df_out.append({'Party': temp_df['Party'].iloc[0], 'Text': working_string, 'UserID': user}, ignore_index=True)
                working_string = ''
                counter += 1
            else: 
                working_string = working_string + ' ' + data
                counter2 += 1
    return df_out

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'Democratic Party':0,'Republican Party':1}

# Dataset class
class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['Party']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['Text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)
    # batch_size = 2 is what produced the best results for us.
    # Increasing batch_size i.e. how many attention head calculations
    # before recalculating embeddings did make the code go faster,
    # but tended to affect performance.
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

# Note on hyperparameters: EPOCHS=5 and LR (learning rate) = 1e-6
# seems to be about right.  More EPOCHS results in overfitting with the
# size of data we were running with.  Increasing learning rate
# makes the results unstable.                  
EPOCHS = 5
model = BertClassifier()
LR = 1e-6

# Evaluate the model using testing data
def evaluate(model, test_data, user_IDs, save_results=False):

    test = Dataset(test_data)
    
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    test_labels = []
    test_prediction = []
    correct_or_no = []
    counter = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            test_labels.append(test_label)
            mask = test_input['attention_mask'].to(device)

            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            if acc == 0:
                correct_or_no.append('no')
            else:
                correct_or_no.append('yes')
            for prediction in output.argmax(dim=1).flatten().tolist():
                test_prediction.append(prediction)
            total_acc_test += acc
            counter += 1
    test_data['UserID'] = user_IDs
    test_data['Predictions'] = test_prediction
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    if not save_results:
        return test_data
    else:
        test_data.to_csv(Path('Data/Raw_BERT_results/testresults.csv'))


def main():

    # First we balance the data set.
    df = pd.read_csv(Path('Data/text_party_IDs.csv'))
    temp_df_dem = df.loc[df['Party'] == 'Democratic Party']
    temp_df_rep = df.loc[df['Party'] == 'Republican Party']
    dem_list = list(set(temp_df_dem['UserID']))
    rep_list = list(set(temp_df_rep['UserID']))
    dem_list = random.sample(dem_list, len(rep_list))

    # Divide training/validation/test sets s.t. each has an equal number of dems and reps. 50/30/20 train/val/test split
    train_dems, test_dems = train_test_split(dem_list, test_size=int(.5*len(dem_list)))
    validation_dems, test_dems = train_test_split(test_dems, test_size=int(.4*len(test_dems)))
    train_reps, test_reps = train_test_split(rep_list, test_size=int(.5*len(rep_list)))
    validation_reps, test_reps = train_test_split(test_reps, test_size=int(.4*len(test_reps)))

    train_users = train_dems + train_reps
    validation_users = validation_dems + validation_reps
    test_users = test_dems + test_reps

    # call the function to construct training/val/test sets
    # on our lists of User IDs corresponding to each.
    df_train = split_into_512_approx(train_users, 7, df)
    df_val = split_into_512_approx(validation_users, 7, df)
    df_test = split_into_512_approx(test_users, 7, df)

    # Drop User IDs, save the test set IDs for evaluation.
    df_train.drop('UserID', axis=1)
    df_val.drop('UserID', axis=1)
    user_ID_add_later = list(df_test['UserID'])
    df_test.drop('UserID', axis=1)

    train(model, df_train, df_val, LR, EPOCHS)
    filename = 'test_trained_model.joblib'
    # serialize and save model
    joblib.dump(model, filename)
    evaluate(model, df_test, user_ID_add_later, save_results=True)
    print(LR, EPOCHS)

if __name__ == "__main__":
    main()
