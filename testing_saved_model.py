import joblib
from nltk.tokenize import word_tokenize
from lemmatize_remove_noise import lemmatize_remove_noise
classifier = joblib.load('trained_classifier')

custom_tweet = "All jokes aside, @tedcruz is a fascist piece of shit."
# lemmatize and remove noise
cleaned_tokens = lemmatize_remove_noise(word_tokenize(custom_tweet))  # ['joke', 'aside', 'tedcruz', 'fascist', 'piece', 'shit']
print(classifier.classify(dict([token, True] for token in cleaned_tokens)))
