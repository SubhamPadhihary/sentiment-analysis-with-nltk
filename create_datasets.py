from nltk import data
import random

def get_tweet_dicts_for_model(cleaned_tweet_tokens):
    '''
    take in a list of cleaned tweet tokens and 
    generates/yields a dictionary required for the naive bayes classifer
    '''
    for tweet_tokens in cleaned_tweet_tokens:
       yield dict([token, True] for token in tweet_tokens)

def get_train_test_data(cleaned_positive_tweet_tokens, cleaned_negative_tweet_tokens):
    positive_dicts = get_tweet_dicts_for_model(cleaned_positive_tweet_tokens)
    negative_dicts = get_tweet_dicts_for_model(cleaned_negative_tweet_tokens)

    # label the datasets
    positive_datasets = [(positive_dict, 'Positive') for positive_dict in positive_dicts]
    negative_datasets = [(negative_dict, 'Negative') for negative_dict in negative_dicts]
    # combine the positive and negative datasets into a single dataset
    dataset = positive_datasets + negative_datasets
    # remove bias by shuffling.
    random.shuffle(dataset)
    # split the data into training and testing in 70:30 ratio
    train_data = dataset[:7000]
    test_data = dataset[7000:]
    return train_data, test_data