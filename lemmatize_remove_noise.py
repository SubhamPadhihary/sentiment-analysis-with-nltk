from nltk.corpus import stopwords
import re, string
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
def lemmatize_remove_noise(tweet_tokens):
    """
    returns a list of cleaned tweet tokens, 
    removing all the urls, twitter handles, punctuation, stop words
    """
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    cleaned_tweet_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        # deal with urls
        token = re.sub('https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '',token)
        # deal with @twitter_handles
        token = re.sub('@[a-zA-Z0-9_]+', '', token)
        # get pos for lemmatize()
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        token = lemmatizer.lemmatize(token, pos)
        # deal with punctuation and stop words, also lowercase.
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tweet_tokens.append(token.lower())
    return cleaned_tweet_tokens
