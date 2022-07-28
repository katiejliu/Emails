#code modified from https://thecleverprogrammer.com/2020/12/20/text-generation-with-python/
#thank you so much!

from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential

# import keras.utils as ku 


from keras.utils import np_utils

from tensorflow.keras.utils import to_categorical

# set seeds for reproducability
# from tensorflow import set_random_seed

from tensorflow.random import set_seed 


from numpy.random import seed
# set_random_seed(2)
set_seed(2)
seed(1)


import pandas as pd
import numpy as np
import string 
import os




curr_file = '/Users/katieliu/Desktop/Emails/Emails.csv'
all_content = []

emails_df = pd.read_csv(curr_file)
all_content.extend(list(emails_df.Content.values))

print(len(all_content))

def clean_text(txt):
    txt="".join(v for v in txt if v not in string.punctuation).lower()
    txt=txt.encode("utf8").decode("ascii",'ignore')
    return txt

corpus = [clean_text(x) for x in all_content]
corpus[:10]

print("reached this part")

tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)

print("reached this part2")

def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    # label = ku.to_categorical(label, num_classes=total_words)
    label = keras.utils.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

print("reached this part3")


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

##part to train the model

# model = create_model(max_sequence_len, total_words)
# model.summary()

# model.fit(predictors, label, epochs=100, verbose=2)

# model.save("model.h5")

##  part to load the model 
from tensorflow.keras.models import load_model

model = load_model('model.h5')
model.summary()

print("reached this part4")


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        # predicted = model.predict_classes(token_list, verbose=0)
        predicted = np.argmax(model.predict(token_list),axis=-1)

        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
   
print (generate_text("love", 50, model, max_sequence_len))
print (generate_text("play", 10, model, max_sequence_len))
print (generate_text("hard work", 4, model, max_sequence_len))
print (generate_text("dad", 4, model, max_sequence_len))
# print (generate_text("new york", 4, model, max_sequence_len))
# print (generate_text("science and technology", 5, model, max_sequence_len))