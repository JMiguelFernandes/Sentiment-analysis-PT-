from nltk.corpus import stopwords
import re
from string import punctuation
from nltk.stem import RSLPStemmer


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