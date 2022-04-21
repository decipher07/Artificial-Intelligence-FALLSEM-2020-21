# %% [markdown]
# # Importing Libraries

# %%
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
%matplotlib inline

import warnings 
warnings.filterwarnings('ignore')

# %% [markdown]
# # Importing Dataset

# %%
df = pd.read_csv("train_tweets.csv")
df.head()

# %%
df.info()

# %% [markdown]
# # Cleaning the Data

# %%
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt);
    for word in r :
        input_txt = re.sub(word, "", input_txt)
    return input_txt

# %%
# remove twitter handles ( @user )
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")

# %%
# remove speical characters, numbers and punctutations
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
df.head()

# %%
# remove short words
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))
df.head()

# %%
# individual words considered as tokens
tokenised_tweet = df['clean_tweet'].apply (lambda x : x.split())
tokenised_tweet.head()

# %% [markdown]
# # Stemming the Words

# %%
# stem the words
# other nlp techniques exists

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

tokenised_tweet = tokenised_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence ])
tokenised_tweet.head()

# %% [markdown]
# # Tokenisation

# %%
#combine words into single sentence 
for i in range (len(tokenised_tweet)):
    tokenised_tweet[i] = " ".join(tokenised_tweet[i])

df ['clean_tweet'] = tokenised_tweet
df.head()

# %% [markdown]
# # Wordcloud for all words

# %%
# visualize the frequent words
all_words = " ".join([sentence for sentence in df['clean_tweet']])

from wordcloud import WordCloud
wordcloud = WordCloud (width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph 
plt.figure (figsize= (15, 8))
plt.imshow (wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# # Wordcloud for Positive Tweet

# %%
# frequent words visualization for +ve

all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label'] == 0]])

wordcloud = WordCloud (width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph 
plt.figure (figsize= (15, 8))
plt.imshow (wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# # WordCloud for Negative Tweets

# %%
# frequent words visualization for -ve

all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label'] == 1]])

wordcloud = WordCloud (width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# plot the graph 
plt.figure (figsize= (15, 8))
plt.imshow (wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %% [markdown]
# # Cleaning Further

# %%
# extract the hashtag 
def hashtag_extract (tweets):
    hashtags = []
    # loop words in the tweets
    for tweet in tweets :
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags


# %%
# extracts hashtags from racist/sexists tweets
ht_positive = hashtag_extract(df['clean_tweet'][df['label']==0])

# extracts hashtags from racist/sexists tweets
ht_negative = hashtag_extract(df['clean_tweet'][df['label']==1])


# %%
ht_positive[:5]


# %%
#unnest lists
ht_positive = sum (ht_positive, [])
ht_negative = sum (ht_negative, [])


# %%
ht_positive[:5]


# %%
freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame ({"Hashtag" : list(freq.keys()), "Count" : list (freq.values())})
d.head()


# %% [markdown]
# # Analysing the Trends

# %%
# select top 10 hashtags 
d = d.nlargest(columns="Count", n = 10)
plt.figure(figsize=(25, 9))
sns.barplot(data=d, x="Hashtag", y="Count")
plt.show()


# %%
freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame ({"Hashtag" : list(freq.keys()), "Count" : list (freq.values())})
d.head()


# %% [markdown]
# # Analysing Top Hashtags Trending

# %%
# select top 10 hashtags 
d = d.nlargest(columns="Count", n = 10)
plt.figure(figsize=(25, 9))
sns.barplot(data=d, x="Hashtag", y="Count")
plt.show()


# %% [markdown]
# # Data Normalization

# %%
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer (max_df=0.90, min_df=2, max_features=1000, stop_words="english")
bow = bow_vectorizer.fit_transform(df['clean_tweet'])
bow


# %% [markdown]
# # Splitting Training and Testing Dataset

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state=42, test_size=0.90)


# %% [markdown]
# # Training the Model

# %%
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train.todense(), y_train)


# %% [markdown]
# # Testing the Model

# %%
y_pred = classifier.predict(x_test.todense())
y_pred


# %% [markdown]
# # Confusion Matrix and Accuracy

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))


# %%
cms=confusion_matrix(y_test,y_pred,labels=[1,0])
sns.heatmap(cms, annot=True, fmt = '.2f')

# %%



