from nltk.corpus import twitter_samples
from lemmatize_remove_noise import lemmatize_remove_noise
from create_datasets import get_train_test_data
from nltk import NaiveBayesClassifier
import joblib
from nltk import classify
if __name__ == '__main__':
    list_of_positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    list_of_negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    cleaned_positive_tweet_tokens = []
    cleaned_negative_tweet_tokens = []
    for tweet_tokens in list_of_positive_tweet_tokens:
        cleaned_positive_tweet_tokens.append(lemmatize_remove_noise(tweet_tokens))
    for tweet_tokens in list_of_negative_tweet_tokens:
        cleaned_negative_tweet_tokens.append(tweet_tokens)
    # print(cleaned_positive_tweet_tokens[0]) # ['#followfriday', 'top', 'engage', 'member', 'community', 'week', ':)']
    # print(cleaned_negative_tweet_tokens[0]) # ['hopeless', 'for', 'tmr', ':(']
    
    # get train and test data
    train_data, test_data = get_train_test_data(cleaned_positive_tweet_tokens, cleaned_negative_tweet_tokens)
    classifier = NaiveBayesClassifier.train(train_data)
    print(classify.accuracy(classifier,test_data))
    print(classifier.show_most_informative_features(10))
    # save model.
    joblib.dump(classifier, 'trained_classifier')