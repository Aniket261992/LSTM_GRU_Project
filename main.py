import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_lstm = load_model('lstm_model.h5')
model_gru = load_model('gru_model.h5')

with open('token.pkl','rb') as f:
    tokenizer = pickle.load(f)

def predict_next_word(tokenizer,model,text,max_seq_len):
    word_list = tokenizer.texts_to_sequences([text])[0]
    if len(word_list)>=max_seq_len:
        word_list = word_list[-(max_seq_len-1):]
    word_list = pad_sequences([word_list],maxlen=max_seq_len,padding='pre')
    predicted_word = model.predict(word_list,verbose=0)
    pred_word_idx = np.argmax(predicted_word,axis=1)

    for word,idx in tokenizer.word_index.items():
        if idx == pred_word_idx:
            return word
        
    return None

st.title("LSTM and GRU Next word predictor")

input_text=st.text_input("Enter the sequence of Words","To be or not to")

if st.button("Predict with LSTM"):
    max_seq_len = model_lstm.input_shape[1]+1
    predicted_word = predict_next_word(tokenizer,model_lstm,input_text,max_seq_len)
    st.write(f'LSTM Next predicted word: {predicted_word}')

if st.button("Predict with GRU"):
    max_seq_len = model_gru.input_shape[1]+1
    predicted_word = predict_next_word(tokenizer,model_gru,input_text,max_seq_len)
    st.write(f'GRU Next predicted word: {predicted_word}')

