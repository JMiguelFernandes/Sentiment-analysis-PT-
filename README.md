# Sentiment-analysis-PT-
A sentiment analysis project for Portuguese (PT) text using a Naive Bayes model trained on tweet data


____


Sentiment analysis tools are commonly used by companies to aid in tasks such as monitoring social media, open survey responses and product reviews. These tools are well developed and widely available for english text, but when it comes to text in other languages the options become more limited. 
For text in Portuguese, for example, (specifically, Portuguese from Portugal) there are no good free publicly available tools for sentiment analysis. One of the reasons for this is the lack of high quality lexicons manually annotated for sentiment polarity, such as those available for English text.


This project is a first attempt at filling this gap. Not a super succesfull one, but a great learning experience nonetheless. :stuck_out_tongue_winking_eye: This Repo is meant as a way to showcase the project and some of the lessons I learned from doing it; use the tools developed here at your own risk ;)

**My approach was to use a corpus made of Tweets, labelled as positive or negative based on the presence of certain keywords, to train a Naive Bayes model to perform a binary classification task. The result was a model that accurately classified around 73% of new tweets as positive or negative**. On top of that, I wrote a small module that can be imported in Python and performs sentiment analysis with just a couple of lines of code in two possible modalities. The first one simply returns a binary sentiment polarity classification (positive or negative). The second shows a breakdown per word of the probabilities of being found in the positive or negative subsets of the training dataset; this allows the user to assess how individual words are contributing to the overall predicted sentiment of a sentence or block of text. 


______
 

## 1. Pros and consof the approach

Cons:

On the down side, the binary allocation of tweets as positive/negative based on keywords is not ideal. The quality of the training dataset is certainly not on the same level as a manually annotated lexicon, with many of the tweets labeled as positive/negative being ambiguous at best. I will go into more detail on this on section 2. Another challenge when dealing with Tweet data is the incredible amount of sarcasm people use there. When you're trying to figure out what sentiment a word carries, sarcastically using it with the exact opposite meaning really throws things out for a loop! Unfortunately, with the very simple approach used here, I had no way of effectively dealing with this, so it probably negatively impacted the quality of the training dataset. I imagine this would also be a problem for any company using sentiment analysis; more sophisticated approaches that include entity recognition and manually annotated datasets, for example, would probably perform much better at identifying sarcasm, but this was out of the scope for this project.

Pros:

However, being able to assemble a training dataset programmatically bypasses the labor and costs associated with developing such a tool. Additionally, the text found on social media or reviews is not the same as literary or wikipedia text, for example; therefore, for analysing text from such sources it could be advantageous to use a tool that has been trained with comparable text, as it could be better at dealing with neologisms and coloquial expressions.


______


## 2. Data collection


To populate the dataset, I collected around 23000 tweets (~16000 positive, ~7000 negative) using a python package called Tweepy, which is a user-friendly wrapper for the Twitter API (check notebook 1). The language parameter in the API search call was set to "pt"; in order to specifically gather tweets in portuguese from Portugal and exclude tweets in Brazilian Portuguese, I also added a geographical restriction using the geocode parameter, consisting of a set of coordinates delimiting a circle that included all of mainland Portugal as well as a chunk of Spain. Some tweets in Brazilian portuguese were still found, but they seemed to be a residual proportion of the dataset.

I chose a few positive and negative keywords more or less arbitrarily (positive: "good", "great", "wonderful", "love"; negative: "horrible", "terrible", "bad", "I hate"). There are probably better ways to choose these keywords, but this provided a quick and easy way to get started. 

| Positive | Negative |
| --- | --- |
| feliz | fml |
| amor | detesto/detestei |
| obrigado/obrigada | trágico/trágica |
| ótimo/ótima | péssimo/péssima |
| parabéns | horrível |
| fantástico/fantástica | terrível |
| maravilha/maravilhoso/maravilhosa | mau/má |

                     
This is one point where the approach could definitely be improved, as some keywords proved less than ideal. Two examples:
- "fml" (as in, "f*** my life") should have been a very specifically negative keyword. Instead, it turns out it is often used in brazilian portuguese as an abbreviation for "família" (family), used coloquially like the english word "fam". So most tweets with this keyword were actually pretty positive-sounding.
- "feliz" (happy) and "bom" (good) resulted in a huge amount of very short tweets like "happy birthday!" or "good day!". Not super useful.

Other than these cases, the tweets collected generally showed the right sentiment (check notebook 2). 

I also found that the timing of the data collection can have a big impact on the dataset quality. Collection would ideally be spread out over a long period of time to avoid over-representation of certain words that just happen to be used a lot over a few days or weeks. For example, football-related words were among the most used words in this dataset. While I would expect this to be the general case for tweets from Portugal (a perpetually football-obsessed country), a lot of the tweets were referring to the final of the Champions League which had just happened on the day before I start data collection. The same was the case for tweets related to Israel/Palestine, as there had been a major escalation in the conflict in that region on the days before. Spreading data collection over a longer period would probably have diluted these effects, so I would definitely recommend doing so.


_______


## 3. Data cleaning and prep

In order to get to a point where I had my text data in a format that could be fed to a machine learning algorithm, some steps of data cleaning and transformation were necessary (check notebook 3). These included:
- removing special characters (Portuguese uses several diacritics, as well as ç).  

     - There are a few words that only differ on the use of diacritics or ç and are otherwise undistinguishable; however, they are very few and I assumed that this wouldn't affect the model significantly.
    
- removing mentions (easy to spot as they start with @), links and emojis

- removing punctuation, newlines, numbers/words containing numbers

- shortening words with the same letter repeated more than once or twice in the case of r and s (rr and ss occur naturally in portuguese; other repeated letters occur only very infrequently).     
    
    - I felt that this was necessary because there were many occurences of words like Yaaaay or adoroooo which just bloat the final term document matrix unnecessarily.
   
- stemming

    - This was done using the portuguese Porter stemmer that's distributed with the nltk library.


After all these steps, the whole corpus was transformed into a format that the algorithm used can handle, namely a document-term matrix. This was a very large object (22842 rows × 12569 columns), so it had to be stored as a sparse matrix rather than a normal pandas dataframe. (Learning about sparse data types and how to handle very large objects was one of the most novel things for me in this project, python-wise :smiley:).


______


## 4. Model training

Many different options exist for model training for sentiment analysis, but I decided to stick with the Naive Bayes classifier for simplicity and performance speed (check notebook 4). This model assumes that all predictors (in this case, the probability of each word appearing in either the positive or negative subset of the training data) are independent of each other and all contribute equally, regardless of the order in which they appear. This is obviously a gross oversimplification, but this sort of model has nevertheless shown high accuracy in many settings such as document classification or spam email filtering. Its greatest advantages are simplicity and scalability (the fact that my laptop can handle it easily was a great plus too :P).

Model training was done using the Multinomial Naive Bayes Classifier from the sklearn library. An 80/20 split between training and validation datasets was used. The model correctly classified ~73% of the tweets in the validation dataset.

A small python module was created that can easily be imported into any python script (just copy the contents of the sentiment_predictor folder into the same folder where your script is located). This module includes the `sentiment_predictor` function, which given a text input returns the sentiment polarity predicted by this model (as well as probabilities per word if the option parameter `extended` is set to True).