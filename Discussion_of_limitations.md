For my own practice, I have decided to write up a quick, informal document detailing my own thoughts and concerns relating to the potential limitations of our model. A less detailed overview of many of the same concerns can be found in the “Reflections” section of the README.

Note: The README belonging to this project was written at a time when I was chunking tweets i.e. concatenating multiple tweets together into one unit for training and classification. I did so to avoid long runtimes and because, at the time, my team and I had reason to believe that we achieved a higher accuracy in that way. Since then, I have retrained my model using individual tweets and noticed an increase in user-level performance. 

## Concerning the generalizability of our model (runtime vs training data):

Our model, as it currently exists, was significantly reduced in scope so as to only consider tweets belonging to US politicians. The model has been both trained and tested exclusively on politicians as of 2/13/2023. Naturally, this limitation makes our model considerably less useful than an idealized version of our model that has been trained and tested on a much larger subsection of the Twitter user base. Regardless, let us speculate as to how well our BERT classifier, in its current state, could generalize to the average twitter user.

We should expect that a much greater share of general population users would be difficult to classify with confidence. Under the assumption that incorrect classifications are balanced (which we will see is unfortunately not the case at the moment), we consider a “confident” classification to be a classification where a high fraction of X tweets belonging to a user share the same tweet-level classification. Many politicians tweet political content almost exclusively, which is not the case for general population twitter users. According to the pew research center (June 16, 2022), one out of every three tweets relates to politics, leaving two thirds that do not (1). As such, we should expect that a large number of tweets for the average user do not contain political messaging, and that whatever classification our system generates for those tweets will likely not be informative. Our plan to combat this issue was to build a second layer of classification that filters out tweets containing no political messaging. The classification process would then be: Use the twitter API to fetch set X of tweets belonging to the subject -> Send X tweets to 1st classifier which will select and return set Y of tweets containing political messaging if Y is not None  -> use Y tweets to generate a prediction for the subject. Were we able to implement this theoretical first classifier, I think it would be a big step towards building a more generalizable system.
The issue with political expression in politicians vs. non-politicians: Let us assume that we were able to build a classifier which was able to accurately filter out non-political content. Even if such a classifier were included in our analysis, and we were able to consider only those tweets which pertained to politics in some way when classifying by party, I expect we would run into issues due to the differences with which politicians and non-politicians express their political views. Intuitively, I would expect political expression in general population users to be more sarcastic, more humorous, less consistent, and potentially more vitriolic. Even if my assumptions here are inaccurate, I think it is fair to say that politicians see twitter, and use twitter, quite differently, and that we should expect a non-trivial difference in the political communications made by politicians and those made by the general user population. As such, I would expect to see a hit in performance if we were to try to generalize the model to non-politicians, even if we were to devise a classifier to sort out non-political tweets. I will keep thinking, but I do not see any means of overcoming this limitation besides training the model on political communications made by labeled, non-politician twitter users.




## Performance over time

Political communication is temporal by nature. As current events change and old topics of conversation are forgotten, we should expect a performance hit unless the classifier was retrained on current data. Our data was retrieved in Oct-Nov of 2022; content which is newer than that, or much older, might confuse our model. For instance, our model does not know anything about the Chinese spy balloon incident in 2023 (2), nor does it know anything about the divided response regarding the Biden administration’s treatment of the situation. 

I hope to conduct a test to get a sense of how well our model will work throughout time as follows:

Attempt to gather 500 tweets from the same subjects for each interval in time: before 2012, before 2014, etc…
Graph raw accuracy and/or some other metrics against time considered.
See if I observe a pattern. I expect to see accuracy peak and we approach the time period from which our data was drawn.


## Risk of classification based on auxiliary characteristics

Our model does not distinguish between political content and non-political content in its current state. There are, of course, tweets which the model was trained on that do not contain political messaging. As such, it is likely that our model is making determinations based on non-political content. This could cause some undesired effects down the line. Let us imagine that our training data contains a disproportionate number of republicans relative to democrats who like to tweet about their dog. Our model then might assume that any tweet pertaining to a dog came from a republican. Naturally, we should be very doubtful that prediction on the basis of “dog” or “no dog” is going to generalize very well. Expanding our training set might take care of this problem to some extent. If it were true that P(Republican|Dog) = P(Democrat|Dog), we should expect to see the share of tweets pertaining to dogs originating from republicans and democrats to approach equality as the size of our training set increases. Our proposed preliminary classifier, which would sort of politically charged tweets before training and testing, would also help to solve this problem.

I do have a question, though. If we did collect more data, and it were surprisingly true that P(Republican|Dog) > P(Democrat|Dog), is there any issue with using this information to inform our predictions? Are there things that could differentiate the communication styles of republicans and democrats on twitter that are not necessarily political in nature? I can think of a few examples that are more believable than the aforementioned dog example:

Reference to religion (3)
Reference to living in urban vs rural areas (4). 
Reference to age (5)

There could be plenty of signal in tweets that are not explicitly political. I doubt that it is possible to make a very accurate determination on non-political communication alone, but it is reasonable to expect that:

P(Republican | lives in NYC, under 30, non-religious) !=  P(Democrat | lives in NYC, under 30, non-religious)

Although I can’t help but wonder what sorts of hard-to-understand patterns of speech might color the social media communications of left-leaning and right-leaning people respectively, there may indeed be a problem here. Especially as more sensitive auxiliary characteristics, race, gender, or sexual orientation for example, are considered, what the system is doing resembles (or just is) inappropriate stereotyping/generalization. 

The harm associated with these generalizations would depend on the use-case. I should think that, if the system were used to target propaganda, the associated harms would be greater than if it was used to, say, generate data for a study in the social sciences. If a model such as ours learned that assigning class X to individuals posting content indicative of their belonging to a particular social group optimized its cost function, those users could be pigeonholed into seeing propaganda that they do not necessarily relate in manner akin to discrimination. Targeted advertisement fueled by machine learning is ethically dubious in general, which doesn’t help, but that is a conversation for another time.

Yet again, I think that an additional layer of classification used to filter out tweets we don’t want to consider could be helpful here. If we were to develop a classifier that was able to label tweets as political or non-political before sending them to our BERT model for party classification, we might be able to prevent our model from being trained on unwanted auxiliary characteristics. Furthermore, if we did decide that we wanted to induce consideration of some auxiliary characteristics, we could do so by adjusting the training set and retraining of the first layer of classification. For instance, we could create a classifier which would sort out tweets containing political content or religious content, and send both to our BERT model for training and classification.


## To chunk or not to chunk

Due to some mistakes and a lack of time to work with when my teammates and I were first working on this model, we decided to chunk tweets together for training and classification. We thought we observed a performance increase and training time decrease by doing this. Since then, I have retrained our model using individual tweets as the training/testing unit. It seems to be true that training on individual tweets takes longer even with the same amount of information, but there is no increase in performance. In fact, training on individual tweets increased my user-level accuracy from 86% to 90%. The source of our error is rather unfortunate. BERT requires that each input consist of the same number of tokens so as to satisfy the transformer architecture of the model. When we did try to train on individual tweets, we forgot to reduce the number of required tokens, so BERT added (many) padding tokens to each of our inputs. This error drastically increased our training time and hurt our performance.

I should think that training and classifying on tweets is more useful, anyway. It should allow us to make predictions even when a user does not have enough data to generate an odd number of chunks, which tended to consist of >10 tweets in our old scheme.  Being able to work with less data could be a big advantage when classifying users who talk politics on twitter infrequently. We also don’t have to worry about the attention heads trying to account for context between tweets where there is none. Furthermore, I think that there may be a statistical advantage to averaging the prediction of many smaller (73 tweets per user on last run) units to generate our user-level predictions as opposed to using fewer large (7 chunks per user on last run) units.


## Additional considerations

As of 2/17/23 I have discovered a few issues with our model that I would like to report.

Issue 1 - Imbalance in misclassification: A few months ago when I was just about to turn this project in, I reported that we were observing equal shares of misclassified republicans and democrats in our test sets. In more recent runs, I have repeatedly observed that it is significantly (approximately x2) more likely for a republican user to be misclassified as a democrat than vice versa. I do not know if the adjustments that I made to the model can account for this shift, or if our observations were due to chance. I construct my (balanced) test, training and validation sets stochastically each run, which could account for this observed inconsistency. I am not sure, at the moment, what is causing this issue; I will have to investigate when I get a chance.


Issue 2 - Failure to classify straightforward user input: I wrote a simple method for generating predictions for custom user input. After playing around with it a bit, I found that my model has a hard time with some very simple inputs. For instance, the phrase “I am a republican” is classified as “Democrat” (ouch). This is just one example: a number of very simple inputs produce unfortunate outputs. In my own experience, as the input gets longer and starts to resemble a cogent, contemporary political argument instead of a simple declaration, the model seems to perform better. I take this to mean that my model is very much in tune with the sort of thing that a politician might tweet, but its capabilities are very narrow. Due to this realization, I consider it even more unlikely that our model would perform well on the general population as it exists now.


(1):
 
https://www.pewresearch.org/politics/2022/06/16/politics-on-twitter-one-third-of-tweets-from-u-s-adults-are-political/


(2): 

https://www.voanews.com/a/republicans-democrats-squabble-over-shoot-down-of-suspected-chinese-spy-balloon/6948797.html

(3):

https://www.pewresearch.org/fact-tank/2016/02/23/u-s-religious-groups-and-their-political-leanings/

(4):

https://en.wikipedia.org/wiki/Urban%E2%80%93rural_political_divide#:~:text=Typically%2C%20urban%20areas%20exhibit%20more,and%2For%20nationalist%20political%20attitudes.

(5):

https://www.pewresearch.org/politics/2020/06/02/the-changing-composition-of-the-electorate-and-partisan-coalitions/


