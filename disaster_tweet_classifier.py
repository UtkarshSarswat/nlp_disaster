import re
import json
import spacy
import pandas as pd
from tqdm import tqdm
import many_stop_words
from autocorrect import Speller
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model
from tensorflow.keras.models import save_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM


class DisasterNewsClassifier():

    def __init__(self, corpus_size):
        self.vector_size = 300
        self.speller_obj = Speller(lang='en')
        self.stop_words = many_stop_words.get_stop_words("en")
        self.spacy_obj = spacy.load('en_core_web_sm')
        self.tokenizer_obj = Tokenizer(num_words=corpus_size, oov_token="<OOV>")
        with open("normalize_mapping.json") as normalize_file_obj:
            self.normalize_mapping = json.load(normalize_file_obj)

    def clean_data(self, each_tweet):
        each_tweet = each_tweet.lower()
        each_tweet = re.sub(r"https?:[^\s]+(?= |$)", "", each_tweet)
        each_tweet = re.sub(r"@[^\s]+", "", each_tweet)
        each_tweet = re.sub(r"&amp;|\||Û|û", "", each_tweet)
        each_tweet = re.sub(r"[^a-z ]", "", each_tweet)
        twitter_slangs = r"|".join(["rt", "tweet", "people"])
        each_tweet = re.sub(twitter_slangs, "", each_tweet)
        each_tweet = self.speller_obj(each_tweet)
        each_tweet = " ".join([each_word.lemma_ for each_word in self.spacy_obj(each_tweet) if each_word not in self.stop_words])
        each_tweet = " ".join([self.normalize_mapping.get(each_word, each_word) for each_word in each_tweet.split()])
        return each_tweet

    def tokenize_sequence(self, train_tweets, val_tweets):
        self.tokenizer_obj.fit_on_texts(train_tweets)
        train_tweets = self.tokenizer_obj.texts_to_sequences(train_tweets)
        val_tweets = self.tokenizer_obj.texts_to_sequences(val_tweets)
        return train_tweets, val_tweets

    def padding_sequence(self, train_tweets, val_tweets):
        train_tweets = pad_sequences(train_tweets)
        max_len = train_tweets.shape[1]
        val_tweets = pad_sequences(val_tweets, maxlen=max_len)
        return train_tweets, val_tweets

    def preprocess_data(self, tweets, labels):
        tweets = list(map(lambda each_tweet: self.clean_data(each_tweet), tqdm(tweets)))
        train_tweets, val_tweets, train_label, val_label = train_test_split(tweets, labels, test_size=0.2, stratify=labels)
        train_tweets, val_tweets = self.tokenize_sequence(train_tweets, val_tweets)
        train_tweets, val_tweets = self.padding_sequence(train_tweets,  val_tweets)
        return train_tweets, val_tweets, train_label, val_label

    def create_model(self, train_tweets):
        input_layer = Input(shape=(train_tweets.shape[1],))
        emb_layer = Embedding(len(self.tokenizer_obj.word_index) + 1, self.vector_size)(input_layer)
        lstm_layer = LSTM(10)(emb_layer)
        output_layer = Dense(1, activation="sigmoid")(lstm_layer)
        class_model = Model(input_layer, output_layer)
        return class_model

    def compile_model(self, model):
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        return model

    def fit_model(self, model, train_tweets, val_tweets, train_label, val_label):
        epochs=10
        result = model.fit(train_tweets, train_label,
            validation_data=(val_tweets, val_label),
            epochs=epochs)
        return result, model

    def train_model(self, train_tweets, val_tweets, train_label, val_label):
        model = self.create_model(train_tweets)
        model = self.compile_model(model)
        result, model = self.fit_model(model, train_tweets, val_tweets, train_label, val_label)

    def main(self, data_file_path, model_file_path):
        data = pd.read_csv(data_file_path)
        train_tweets, val_tweets, train_label, val_label = self.preprocess_data(data["text"], data["target"])
        result, model = self.train_model(train_tweets, val_tweets, train_label, val_label)
        save_model(model, model_file_path)


if __name__=='__main__':
    disaster_tweet_class_obj = DisasterNewsClassifier(20000)
    disaster_tweet_class_obj.main("train.csv", "nlp_disaster")

