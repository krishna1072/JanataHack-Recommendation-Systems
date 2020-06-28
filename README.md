## JanataHack-Recommendation-Systems
JanataHack: Recommendation Systems

https://datahack.analyticsvidhya.com/contest/janatahack-recommendation-systems/#ProblemStatement

# Problem Statement:
Your client is a fast-growing mobile platform, for hosting coding challenges. They have a unique business model, where they crowdsource problems from various creators(authors). These authors create the problem and release it on the client's platform. The users then select the challenges they want to solve. The authors make money based on the level of difficulty of their problems and how many users take up their challenge.

The client, on the other hand makes money when the users can find challenges of their interest and continue to stay on the platform. Till date, the client has relied on its domain expertise, user interface and experience with user behaviour to suggest the problems a user might be interested in. You have now been appointed as the data scientist who needs to come up with the algorithm to keep the users engaged on the platform.

The client has provided you with history of last 10 challenges the user has solved, and you need to predict which might be the next 3 challenges the user might be interested to solve. Apply your data science skills to help the client make a big mark in their user engagements/revenue.

# Solution
For each user in the training set create 3 observations: sequence of 10 challenges solved and three labels (for 11th, 12th and 13th challenges). So, now we have a multiclassification problem with 5k classes and about 200k observations. The classification is done using a single Recurrent Neural Net with BiDirectional LSTM layer.

During test time, obtain a probability distribution for each sequence. Then, choose top-3 argmax probabilities as 11th, 12th and 13th challenges predicted.
