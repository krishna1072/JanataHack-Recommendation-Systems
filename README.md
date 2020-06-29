## JanataHack-Recommendation-Systems
JanataHack: Recommendation Systems

https://datahack.analyticsvidhya.com/contest/janatahack-recommendation-systems/#ProblemStatement

# Problem Statement:
The client has provided you with history of last 10 challenges the user has solved, and you need to predict which might be the next 3 challenges the user might be interested to solve.

## Approach:
# RNN_Recommendation_Systems.ipynb
1. For each user in the training set create 3 observations: sequence of 10 challenges solved and three labels (for 11th, 12th and 13th challenges)
2. Thus we have a multiclassification problem with 5k classes and about 200k observations. The classification is done using a single Recurrent Neural Net with BiDirectional LSTM layer
3. During test time, obtain a probability distribution for each sequence. Then, choose top-3 argmax probabilities as 11th, 12th and 13th challenges predicted

# NN_Recommendation_Systems.ipynb
1. Similar to sequence of words used to predict the next word (like in time series) or similar words (like in market basket analysis)
2. Built embeddings for the challenges using gensim word2vec.
3. The embeddings were used to find the most similar words/challenges to the last few challenges the user has attempted.
4. This model was optimsed first. The size of embedding, iteration and window size was tuned
5. Built a GRU network which predicts the next challenge given the 10 challenges a user has solved
6. The target variable was the list of challenges solved on 11th,12th,13th place
7. The train data samples were duplicated 3 times ( 1 each for 11th, 12th, 13th)
8. Sample weights were given to each sample and the samples in which target was 11th challenge, were given more importance
9. The word embeddings trained from word2vec were fed into the GRU.
10. The GRU was trained for different sets of training data
11. Predcitions were made by taking the most 3 likely samples for each user
