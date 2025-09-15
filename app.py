import streamlit  as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle 
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
import numpy as np
from sklearn.model_selection import train_test_split
import  pickle 
 
model=load_model('model_word_lstm.h5')
with open('tokenizer.pickle','rb') as file: # Feature engineered Column Geography
    tokenizer=pickle.load(file)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list [-(max_sequence_len-1):]
        # ensure the sequence length match max sequence len - 1
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
st.title("Word Prediction Project ")
input_name = st.text_input('Enter your name  ')
input_text  = st.text_input('Enter your text  ')

print(f" Input Text is : {input_text}")

max_sequence_len = model.input_shape[1]+1

next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
if input_name:
    st.write(f"HI {input_name} ,if u want me to predict next word enter word or sentence")
if input_text:
    st.success(f"Next Word Prediction: {next_word}")
