import pandas as pd
import numpy as np
import streamlit as st
import torch

import snscrape.modules.twitter as sntwitter
import datetime
import pickle 

import warnings
import account_classifier as ac

warnings.filterwarnings('ignore')

st.set_page_config(page_title="App Name")

st.title("TUTI - Tweet Buzzer Detection")

tab1, tab2, tab3 = st.tabs(["Tweet Buzzer Detection", "Account Buzzer Detection", "Tweet Conversation"])

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


with tab1:
    #input text
    sentence = st.text_input('Input Sentence:') 

    #transform/Extract (TF-IDF) new text 
    text_new =[sentence]
    X_new_counts = count_vect.transform(text_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    #Predict new text
    prediction = classifier.predict(X_new_tfidf)
    prediction_proba = classifier.predict_proba(X_new_tfidf)

    # 1 buzzer, 2 non buzzer
    label = "Tweet ini kemungkinan sedang mempromosikan atau membentuk opini publik." if prediction_proba[0][1] >= 0.7 else "Tweet ini tidak mencoba mempengaruhi opini publik secara berlebihan."
    #label = f"{label} ({prediction_proba[0][1]})"

    if sentence:
        st.subheader("Hasil prediksi:")
        st.write(label) 

with tab2:
    #input text
    akun = st.text_input('Input Account:')
    
    if akun:
        # get tweets and account details
        accounts =[akun]

        raw_users, raw_tweets = ac.get_tweets_new_format(accounts)
        processed_data = ac.pre_process(raw_users, raw_tweets)
        
        
        #Predict new text
        p = ac.predict(gnn, processed_data)

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

        st.subheader("Hasil prediksi:")
        st.write(label)
        st.write(stats)

#Tweet Conversation Feature
with tab3:
    st.header("Tweet Conversation")
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
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_term + ' since:'+since_date+' until:'+formatted_date).get_items()):
                if i>limit-1:
                    break
                tweets_list2.append([tweet.content, tweet.user.username])
                
            # Creating a dataframe from the tweets list above
            tweets_df2 = pd.DataFrame(tweets_list2, columns=['Text', 'Username'])
            
            X_new_counts = count_vect.transform(tweets_df2['Text'])
            X_new_tfidf = tfidf_transformer.transform(X_new_counts)

            #Predict new text
            prediction = classifier.predict(X_new_tfidf)
            prediction_proba = classifier.predict_proba(X_new_tfidf)

            prediction_df = pd.DataFrame({'Predicted': prediction, 'Prediction_Probability': prediction_proba[:,1]})
            # Concatenate the new DataFrame with the original DataFrame
            tweets_df_with_pred = pd.concat([tweets_df2, prediction_df], axis=1)
            tweets_df_with_pred = tweets_df_with_pred[tweets_df_with_pred['Predicted']==0]
            tweets_df_with_pred = tweets_df_with_pred[['Text','Username']]
            tweets_df_with_pred = tweets_df_with_pred.reset_index().drop(columns='index')
            #st.table(tweets_df_with_pred)
            #pd.set_option('display.max_rows', None)
            # CSS to inject contained in a string
            hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
            if tweets_df_with_pred.empty:
                st.write("Topik ini kemungkinan besar mengandung Pembentukan Opini publik")
            else:
                st.table(tweets_df_with_pred)