import pandas as pd
import streamlit as st

import pickle 

st.set_page_config(page_title="App Name")

st.title("App Name")
st.header("App Name")

#df = pd.read_csv('dummyhatespeech.csv')
pkl_filename = "LR_Model.pkl"

#load model pickle
with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)

#load TF Weight
with open("countVectLR", 'rb') as file:
    count_vect = pickle.load(file)

#load IDF Weight
with open("tfidfLR", 'rb') as file:
    tfidf_transformer = pickle.load(file)

#input text
sentence = st.text_input('Masukkan Kalimat:') 

#transform/Extract (TF-IDF) new text 
text_new =[sentence]
X_new_counts = count_vect.transform(text_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#Predict new text
prediction = classifier.predict(X_new_tfidf)
prediction_proba = classifier.predict_proba(X_new_tfidf)

if sentence:
    st.subheader("Hasil prediksi:")
    st.write(prediction[0])
          
          
          
        

    
