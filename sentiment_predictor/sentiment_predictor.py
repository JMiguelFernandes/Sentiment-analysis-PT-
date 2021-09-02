from nltk.corpus import stopwords
import re
from string import punctuation
from nltk.stem import RSLPStemmer

import pandas as pd
import numpy as np
import pickle
from math import e

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix



def remove_pt_special_chars(text):
    '''Replaces portuguese special characters (accented vowels and ç) with their base cognates. Input should be lower case.'''
    text = re.sub(r"[àáãâ]", "a", text)
    text = re.sub(r"[éê]", "e", text)
    text = re.sub(r"[í]", "i", text)
    text = re.sub(r"[óôõ]", "o", text)
    text = re.sub(r"[ú]", "u", text)
    text = re.sub(r"[ç]", "c", text)
    return(text)
    
    
def remove_mentions(text):
    '''Removes mentions to other twitter users.'''
    text = re.sub(r"(@[^ !]*)", "", text)
    return(text)
    
    
def remove_links(text):
    '''Removes links from text.'''
    text = re.sub(r"http[\S]*", "", text)
    text = re.sub(r"www.[\S]*", "", text)
    return(text)
    
    
def clean_punctuation_and_newlines(text):
    '''Removes punctuation *except* exclamation marks. Reduces multiple exclamation marks to a single one and adds a space 
    between words and exclamation mark. Also removes newline characters and reduces multiple spaces to one'''
    text = re.sub(f"[{punctuation[1:]}“”]", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"(!+)", " ! ", text)
    text = re.sub(r" +", " ", text)
    return(text)
    
    
def remove_numbers(text):
    '''Removes numbers and words containing numbers'''
    text = re.sub(r"\S*[0-9]+\S*", " ", text)
    return(text)
    
    
def remove_emojis(text):
    '''Removes most emojis from the text; some are still left.'''
    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    return RE_EMOJI.sub(r'', text)
    
    
def remove_repeated_letters(text):
    '''Reduces repeated letters except r or s anywhere in each word by a single letter. Repeated r or s inside words are 
    reduced to rr or ss respectively, to account for words where this naturally occurs. Repeated r or s at the end of words are
    reduced to single letters.
    '''
    
    clean_text = []
    for i in text.split():
        word = re.sub(r"([a-z])\1{1,}$", r"\1", i)         #replaces repeated letters at the end of the word with single letter
        word = re.sub(r"([^sr])\1{1,}(?=.)", r"\1", word)  #replaces repeated letters (except r or s) anywhere with single letter
        word = re.sub(r"([rs])\1{1,}(?=.)", r"\1\1", word) #replaces repeated internal r or s with double r or s
        clean_text.append(word)
    return(" ".join(clean_text))
    
    
def apply_stemming(text):    
    '''Applies stemming using the Porter stemmer implementation for portuguese contained in the nltk library.
    Still not sure if it's worth using this or not, because the results are... sketchy, to say the least.
    '''
    stemmer = RSLPStemmer()
    return(" ".join([stemmer.stem(i) for i in text.split()]))
    
    
def clean_up_tweets(tweet):
    '''Applies all the above functions sequentially to turn a piece of raw text into a format that is more
    appropriate for machine learning models to handle. Will still contain stopwords (see remove_stopwords)
    '''
    clean_tweet = tweet.lower()
    clean_tweet = remove_mentions(clean_tweet)
    clean_tweet = remove_links(clean_tweet)
    clean_tweet = remove_numbers(clean_tweet)
    clean_tweet = remove_emojis(clean_tweet)
    clean_tweet = clean_punctuation_and_newlines(clean_tweet)
    clean_tweet = apply_stemming(clean_tweet)
    clean_tweet = remove_pt_special_chars(clean_tweet) 
    clean_tweet = remove_repeated_letters(clean_tweet)

    return(clean_tweet.strip(" "))
    
    
processed_stopwords = [clean_up_tweets(i) for i in stopwords.words("portuguese")] + ["q", "k", "c", "p"]
processed_stopwords.remove("nao")

positive_keywords = ["feliz",
                     "amor",
                     "obrigado OR obrigada",
                     "ótimo OR ótima",
                     "parabéns",
                     "fantástico OR fantástica", 
                     "maravilha OR maravilhoso OR maravilhosa"]
            
                     
negative_keywords = ["fml",
                     "péssimo OR péssima",
                     "trágico OR trágica",
                     "horrível",
                     "mau OR má",
                     "terrível", 
                     "detesto OR detestei"]

processed_keywords = []
for i in positive_keywords + negative_keywords:
    for j in i.split():
        if j != "OR":
            processed_keywords.append(clean_up_tweets(j))
                        
processed_stopwords = list(set(processed_stopwords + processed_keywords))

def remove_stopwords(text, stopwords):
    '''Removes stopwords from text (assuming the text has already been cleaned up with the clean_up_tweets
    function first)
    '''
    return(" ".join([i for i in text.split() if i not in stopwords]))




infile = open("sentiment_predictor/model_multinomialNB", "rb")
model_multinomialNB = pickle.load(infile)
infile.close()

infile = open("sentiment_predictor/feature_log_probs_df", "rb")
feature_log_probs_df = pickle.load(infile)
infile.close()

infile = open("sentiment_predictor/vocabulary", "rb")
vocabulary = pickle.load(infile)
infile.close()




def sentiment_predictor(text, extended = False):
    """This function returns the sentiment polarity predicted by a Naive Bayes model trained on a dataset of 
    tweets. If extended is set to True, it also returns the probabilities per word of being in the negative 
    or positive subsets of the training dataset.
    
    Notes on the extended results: 
    
    Some words that are present in the input text might not appear in the extended results table. This 
    is due to one of three reasons:
    - the word is one of the keywords used in building the training dataset
    - the word is a stopword (a list of words in portuguese that are so common that they don't contribute 
    much meaning to the sentiment analysis)
    - the word was not present in the training dataset
    
    Sometimes, the average of the probabilities in the extended results table would suggest that the text 
    should be considered negative but the result is positive. This is due to the fact that the training
    dataset was imbalanced, containing more positive tweets than negative ones. This ends up favoring
    positive classifications in edge cases.
    """
    processed_text = remove_stopwords(clean_up_tweets(text), processed_stopwords)
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform([processed_text])
    words = count_vectorizer.get_feature_names()

    text_vector = [1 if i in words else 0 for i in vocabulary]
    prediction = model_multinomialNB.predict([text_vector])[0]
    
    if not extended:

        if prediction == 0:
            print("Negative")
            return(0)
        else:
            print("Positive")
            return(1)
    else:
        
        if prediction == 0:
            print("Negative")
        else:
            print("Positive")

        words = []
        neg_probs = []
        pos_probs = []

        for word in processed_text.split(" "):
            words.append(word)
            neg_probs.append(feature_log_probs_df.loc[word, "p_negative"])
            pos_probs.append(feature_log_probs_df.loc[word, "p_positive"])

        word_probs_df = (pd.DataFrame([words, neg_probs, pos_probs])
                         .transpose()
                         .rename(columns={0:"processed word", 1:"p_negative", 2:"p_positive"})
                         .set_index("processed word"))
        return(word_probs_df)