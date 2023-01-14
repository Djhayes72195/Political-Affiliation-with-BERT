# Data cleaning for model fitting
import os
# os.chdir('/Users/dustinhayes/Desktop/STAT766Final')
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
from sklearn.model_selection import train_test_split



#You will have to change the path on the next line.  It should point to wherever you put the
#dataset I gave you.
df = pd.read_csv('text_party_IDs.csv')
temp_df_dem = df.loc[df['Party'] == 'Democratic Party']
temp_df_rep = df.loc[df['Party'] == 'Republican Party']
dem_list = list(set(temp_df_dem['UserID']))
rep_list = list(set(temp_df_rep['UserID']))
dem_list = random.sample(dem_list, len(rep_list))
user_list = rep_list + dem_list
# Next line JUST for testing
# user_list = random.sample(user_list, 30)
train_users, test_users = train_test_split(user_list, test_size=int(.5*len(user_list)))
test_users, validation_users = train_test_split(test_users, test_size=int(.4*len(test_users)))



counter2 = 0
def split_into_512_approx(users, tweet_blocks_per_user, no_of_tweets=0, chunk_by_length=True):
    """This function will take in a list of users and construct either test
    or train dataframe.  If chunk by length you should specify how many tweet
    blocks you want per user.  If not, you should specify how many tweets you want from
    each user with no_of_tweets. In each case, the longest tweets will be picked first, on the
    assumption that longer tweets would be more likely to contain political messaging.
    I'm not sure why, but you get a few more tweets than what you actually
    put in the argument.  

    This function was constructed so as to ensure that tweets from any particular
    politician only appear in only the test or training set, but not both.
    """
    df_out = pd.DataFrame(columns=['Party', 'Text', 'UserID'])
    num_dem = 0
    num_rep = 0
    all_info = df
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
            if chunk_by_length:
                if counter >= tweet_blocks_per_user:
                    break
                if len((working_string + ' ' + data).split(' ')) > 500:
                    df_out = df_out.append({'Party': temp_df['Party'].iloc[0], 'Text': working_string, 'UserID': user}, ignore_index=True)
                    working_string = ''
                    counter += 1
                else: 
                    working_string = working_string + ' ' + data
                    counter2 += 1
            else:
                if counter >= (no_of_tweets):
                    break
                else:
                    df_out = df_out.append({'Party': temp_df['Party'].iloc[0], 'Text': data, 'UserID': user}, ignore_index=True)
                    counter += 1
                    counter2 += 1
    print(counter2)
    print(num_dem, num_rep)
    return df_out

# Dataset class
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'Democratic Party':0,'Republican Party':1}


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

# split the dataframe into training, validation, and test set
# np.random.seed(115)
# df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])
# print(len(df_train),len(df_val), len(df_test))
# # BERT model
# new method of creating test/train dfs s.t. any particular pol only appears in one
df_train = split_into_512_approx(train_users, 7)
df_val = split_into_512_approx(validation_users, 7)
df_test = split_into_512_approx(test_users, 7)
# df_val.to_csv('/Users/dustinhayes/Desktop/STAT766Final/BERT_results/val_confirm.csv')
# df_train.to_csv('/Users/dustinhayes/Desktop/STAT766Final/BERT_results/df_train_confirm.csv')
# df_test.to_csv('/Users/dustinhayes/Desktop/STAT766Final/BERT_results/test_confirm.csv')
print(df_val)
print(df_test)
print('df_test={}'.format(df_test))
df_train.drop('UserID', axis=1)
df_val.drop('UserID', axis=1)
user_ID_add_later = list(df_test['UserID'])
df_test.drop('UserID', axis=1)


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        #I changed the 5 in the next one to 2 for testing
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

# train BERT model
#@jit(target_backend='cuda')
def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

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
                  
EPOCHS = 5
model = BertClassifier()
LR = 1e-6

#@jit(target_backend='cuda')
#def trainv2():
train(model, df_train, df_val, LR, EPOCHS)

#trainv2()

# Evaluate the model using testing data
def evaluate(model, test_data):
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
            print(output.argmax(dim=1))
            # test_prediction.append(output.argmax(dim=1))
            for prediction in output.argmax(dim=1).flatten().tolist():
                test_prediction.append(prediction)
            total_acc_test += acc
            counter += 1
    df_test['UserID'] = user_ID_add_later
    df_test['Predictions'] = test_prediction
    df_test.drop(columns=['Text'])
    df_test.to_csv('BERT_results/testresults_good_split.csv')
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
    
evaluate(model, df_test)
print(LR, EPOCHS)
