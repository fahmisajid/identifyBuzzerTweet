import pandas as pd
import streamlit as st

import pickle 

st.set_page_config(page_title="Tuti Tweet Buzzer Detector v0.1")

st.title("Tuti Tweet Buzzer Detector v0.1")
st.header("Text classifier")

#df = pd.read_csv('dummyhatespeech.csv')
pkl_filename = "buzzermodel"

#load model pickle
with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)

#load TF Weight
with open("countvect", 'rb') as file:
    count_vect = pickle.load(file)

#load IDF Weight
with open("tfidf", 'rb') as file:
    tfidf_transformer = pickle.load(file)

#input text
sentence = st.text_input('Masukkan teks:') 

#transform/Extract (TF-IDF) new text 
text_new =[sentence]
X_new_counts = count_vect.transform(text_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#Predict new text
prediction = classifier.predict(X_new_tfidf)
prediction_proba = classifier.predict_proba(X_new_tfidf)

# 1 buzzer, 2 non buzzer
label = "Buzzer" if prediction[0] == 1 else "Non Buzzer"
label = f"{label} ({prediction_proba[0][1]})"
if sentence:
    st.subheader("Hasil prediksi:")
    st.write(label)
          
          
          
        

    
