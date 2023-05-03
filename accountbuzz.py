import pandas as pd
import numpy as np
import streamlit as st
import torch
import time
#import snscrape.modules.twitter as sntwitter
import datetime
import pickle 

import warnings
import account_classifier as ac

st.set_page_config(page_title="App Name")

st.title("TUTI - Tweet Buzzer Detection")

tab1, tab2, tab3 = st.tabs(["Tweet Buzzer Detection", "Account Buzzer Detection", "Tweet Conversation"]) #define tab feature

#Inisisasi Model (LOAD PICKLE)
#==============================================================================================================
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

# load GNN model
GNN_PATH = "model_new_format_v2.pt"
gnn = torch.load(GNN_PATH)
gnn.eval()

#Fitur
#==============================================================================================================
#tab1 - Tweet Buzzer Detection
with tab1:
    #input text
    sentence = st.text_input('Masukkan Opini:') 

    predict_text_start = time.time()
    #transform/Extract (TF-IDF) new text 
    text_new =[sentence]
    X_new_counts = count_vect.transform(text_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    #Predict new text
    prediction = classifier.predict(X_new_tfidf)
    prediction_proba = classifier.predict_proba(X_new_tfidf)
    predict_text_time = time.time() - predict_text_start

    # 1 buzzer, 2 non buzzer
    label = "Tweet ini kemungkinan sedang mempromosikan atau membentuk opini publik." if prediction_proba[0][1] >= 0.7 else "Tweet ini tidak mencoba mempengaruhi opini publik secara berlebihan."
    #label = f"{label} ({prediction_proba[0][1]})"


    if sentence:
        if len(sentence) > 280:
            st.warning('Peringatan: Jumlah kata yang kamu inputkan melebihi batas karakter maksimal 280 kata. Silakan periksa kembali teks yang kamu masukkan.', icon="⚠️")
        elif len(sentence.split()) < 2:
            st.warning("Peringatan: Kalimat setidaknya memiliki 2 kata", icon="⚠️")
        else:
            st.subheader("Hasil prediksi:")
            st.write(label)
            st.subheader("Waktu prediksi:")
            st.write(f"{round(predict_text_time, 3)}s")

#tab2 - Accpunt Buzzer Detection
with tab2:
    get_tweet_time = 0
    predict_time = 0
    #Define Matrix Info for an Account
    col21, col22, col23 = st.columns(3)
    col24, col25, col26 = st.columns(3)

    #input Account
    akun = ""
    akun = st.text_input('Masukkan Akun:')  
    if akun:
        # get tweets and account details

        accounts =[akun]
        try:
            start_get_tweet = time.time()
            raw_users, raw_tweets = ac.get_tweets_new_format(accounts)
            get_tweet_time = time.time() - start_get_tweet
        except:
            st.warning("Maaf, akun yang Anda masukkan tidak ditemukan. Pastikan akun yang dimasukkan benar atau cek kembali penulisan username-nya.", icon="🚨")
        
        start_predict = time.time()
        processed_data = ac.pre_process(raw_users, raw_tweets)
        
        
        #Predict new text
        p = ac.predict(gnn, processed_data)
        predict_time =  time.time() - start_predict

        buzzer = p[0].item() == 1

        # 1 buzzer, 0 non buzzer
        location = raw_users[0]["location"]
        created_at = raw_users[0]["created_at"]
        protected = raw_users[0]["protected"]
        favourites_count = raw_users[0]["favourites_count"]
        followers_count = raw_users[0]["followers_count"]
        friends_count = raw_users[0]["friends_count"]
        verified = "True" if raw_users[0]["verified"] == 1 else "False"
        statuses_count = raw_users[0]["statuses_count"]
        label = "Akun ini kemungkinan mempromosikan atau membentuk opini publik." if buzzer >= 1 else "Akun ini tidak mencoba mempengaruhi opini publik secara berlebihan."
        stats = f"Verified: {verified} | Followers: {followers_count} | Following: {friends_count} | Location: {location} | Since: {created_at} | Favourites: {favourites_count} | Tweets: {statuses_count}"
        #label = f"{label} ({prediction_proba[0][1]})"

        date_list = created_at.split()
        month = date_list[1]
        day = date_list[2]
        year = date_list[-1]
        datecreated = day + " " + month + " " + year
        col21.metric("Nama Akun", akun)
        col22.metric("Verified", verified)
        col23.metric("Lokasi", location)
        col24.metric("Jumlah Pengikut", followers_count)
        col25.metric("Jumlah yang diikuti", friends_count)
        col26.metric("Tanggal Akun dibuat", datecreated)
        st.subheader("Hasil prediksi:")
        st.write(label)
        st.subheader("Waktu prediksi:")
        st.write(f"{round(get_tweet_time, 3)}s for getting tweets and {round(predict_time, 3)}s for prediction")
        #st.write(stats)

        tweets = ac.get_flattened_tweets(raw_tweets)[["full_text", "is_retweet", "n_video_media", "n_photo_media",	"retweet_count","favorite_count",	"replies"]]
        st.table(tweets)

        akun = ""
#
with tab3:
    col1, col2, col3 = st.columns(3)
    opsi = st.radio(
    "Pilih metode pencarian tweet conversation",
    ('Cari dengan kata kunci', 
     'Upload file CSV untuk pencarian'))

    
    if opsi == 'Cari dengan kata kunci':
        try:
            with st.form(key='Twitter_form'):
                search_term = st.text_input('What topic do you want to search for?')
                #limit = st.slider('How many tweets do you want to get?',1,100)
                limit = 100
                #day_before = st.slider("What is the number of days before today that you want to retrieve the tweet from?",1,30)
                day_before = 1
                current_date = datetime.date.today()
                formatted_date = current_date.strftime('%Y-%m-%d')
                previous_date = current_date - datetime.timedelta(days=day_before)
                since_date = previous_date.strftime('%Y-%m-%d')

                submit_button = st.form_submit_button('Submit')
                

                if submit_button:
                    tweets_list2 = []
                    # Using TwitterSearchScraper to scrape data and append tweets to list
                    raise TypeError("Scrapping Lib still Error")    
                    # Creating a dataframe from the tweets list above
                    tweets_df2 = pd.DataFrame(tweets_list2, columns=['Text', 'Username'])
                    
                    X_new_counts = count_vect.transform(dataframe['Text'])
                    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

                    #Predict new text
                    prediction = classifier.predict(X_new_tfidf)
                    prediction_proba = classifier.predict_proba(X_new_tfidf)

                    prediction_df = pd.DataFrame({'Predicted': prediction, 'Prediction_Probability': prediction_proba[:,1]})
                    # Concatenate the new DataFrame with the original DataFrame
                    tweets_df_with_pred = pd.concat([dataframe, prediction_df], axis=1)
                    tweets_buzzer = tweets_df_with_pred[tweets_df_with_pred['Predicted']==1]
                    tweets_Genuine = tweets_df_with_pred[tweets_df_with_pred['Predicted']==0]
                    tweets_Genuine = tweets_Genuine.reset_index().drop(columns='index')
                    #tweets_df_with_pred = tweets_df_with_pred[tweets_df_with_pred['Predicted']==0]
                    tweets_df_with_pred = tweets_df_with_pred[['Text','Username']]
                    tweets_df_with_pred = tweets_df_with_pred.reset_index().drop(columns='index')
                    #st.table(tweets_df_with_pred)
                    #pd.set_option('display.max_rows', None)
                    # CSS to inject contained in a string
                    if tweets_df_with_pred.empty:
                        st.write("Topik ini kemungkinan besar mengandung Pembentukan Opini publik")
                    else:
                        st.table(tweets_Genuine[['Text','Username']])

                    col1.metric("Talker", tweets_df_with_pred.shape[0])
                    col2.metric("Buzzer", tweets_buzzer.shape[0])
                    col3.metric("Genuine", tweets_Genuine.shape[0])

        except:
            st.warning("Mohon maaf, kami mengalami kendala saat melakukan crawling pada tweet conversation, kami akan segera memperbaikinya. Silakan coba lagi nanti.", icon="🚨")

    elif opsi == 'Upload file CSV untuk pencarian':
        uploaded_file = st.file_uploader("Choose a file") 
        #col4 = st.columns(1)
        if uploaded_file is not None:
            predict_file_start = time.time()
            dataframe = pd.read_csv(uploaded_file)
            st.write("filename:", uploaded_file.name)
            

            X_new_counts = count_vect.transform(dataframe['Text'])
            X_new_tfidf = tfidf_transformer.transform(X_new_counts)

            #Predict new text
            prediction = classifier.predict(X_new_tfidf)
            prediction_proba = classifier.predict_proba(X_new_tfidf)

            prediction_df = pd.DataFrame({'Predicted': prediction, 'Prediction_Probability': prediction_proba[:,1]})
            # Concatenate the new DataFrame with the original DataFrame
            tweets_df_with_pred = pd.concat([dataframe, prediction_df], axis=1)
            tweets_df_with_pred["Detected as Buzzer"] = tweets_df_with_pred["Predicted"].apply(lambda x: "Yes" if x == 1 else "No")
            tweets_buzzer = tweets_df_with_pred[tweets_df_with_pred['Predicted']==1]
            tweets_Genuine = tweets_df_with_pred[tweets_df_with_pred['Predicted']==0]
            
            predict_file_time = time.time() - predict_file_start
            tweets_Genuine = tweets_Genuine.reset_index().drop(columns='index')
            #tweets_df_with_pred = tweets_df_with_pred[tweets_df_with_pred['Predicted']==0]
            tweets_df_with_pred = tweets_df_with_pred[['Text','Username']]
            tweets_df_with_pred = tweets_df_with_pred.reset_index().drop(columns='index')

            st.subheader("Waktu prediksi:")
            st.write(f"{round(predict_file_time, 3)}s")
            
            st.table(tweets_Genuine[['Text','Username', 'Detected as Buzzer']])

            if tweets_df_with_pred.empty:
                st.write("Topik ini kemungkinan besar mengandung Pembentukan Opini publik")
            else:
                st.table(tweets_buzzer[['Text','Username', 'Detected as Buzzer']])

            col1.metric("Talker", tweets_df_with_pred.shape[0])
            col2.metric("Buzzer", tweets_buzzer.shape[0])
            col3.metric("Genuine", tweets_Genuine.shape[0])                                                                   
