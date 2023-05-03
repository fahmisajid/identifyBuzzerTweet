from twitter_scraper_selenium import scrape_profile
import torch
import pandas as pd

from datetime import datetime
def pre_process(raw_users, raw_tweets, with_label=False):
  flattened_tweets = []
  for _id in raw_tweets:
    r = raw_tweets[_id]
    flattened_tweets.append(dict(
        id=_id,
        username=r["username"],
        full_text=r["content"],
        user_mentions=r["mentions"],
        n_video_media=len(r["videos"]),
        n_photo_media=len(r["images"]),
        retweet_count=r["retweets"],
        favorite_count=r["likes"],
        replies=r["replies"],
        is_retweet = 1 if r["is_retweet"] else 0
    ))
  df_tweets = pd.DataFrame(flattened_tweets)

  f = "%a %b %d %H:%M:%S %z %Y"
  flattened_users = []
  for r in raw_users:
    if r is not None:
      flattened_users.append(dict(
          id= r["id"],
          username=r["screen_name"],
          name= r["name"],
          screen_name= r["screen_name"],
          location= r["location"],
          profile_location= r["profile_location"],
          description= r["description"],
          # account_age= ( datetime.now(timezone.utc) - datetime.strptime(r["created_at"], f)).days,
          account_age= (datetime.strptime("2023-03-05 00:00:00+07:00", "%Y-%m-%d %H:%M:%S%z") - datetime.strptime(r["created_at"], f)).days,
          created_at= datetime.strptime(r["created_at"], f),
          geo_enabled= int(r["geo_enabled"]),
          protected= int(r["protected"]),
          favourites_count= r["favourites_count"],
          followers_count= r["followers_count"],
          friends_count= r["friends_count"],
          listed_count= r["listed_count"],
          verified= int(r["verified"]),
          statuses_count= r["statuses_count"],
          default_profile= int(r["default_profile"]),
          default_profile_image= int(r["default_profile_image"]),
          has_extended_profile= int(r["has_extended_profile"]),
          Y= r["Y"] if with_label else None
      ))
  df_users = pd.DataFrame(flattened_users)

    # df_merged = pd.merge(df_tweets, df_users, how='left', left_on='user_id', right_on='id')

  import torch
  import torch_geometric.transforms as T

  def get_graph(username):
    # Sort to define the order of nodes
    df_sorted_user = df_users[df_users["username"] == username].sort_values(by="username").set_index("username")
    df_sorted_user = df_sorted_user.reset_index(drop=False)
    user_id_mapping = df_sorted_user["username"]

    user_node_features = df_sorted_user[["account_age","geo_enabled", "protected",	"favourites_count",	"followers_count",	"friends_count",	"listed_count",	"verified",	"statuses_count",	"default_profile",	"default_profile_image",	"has_extended_profile"]]

    df_sorted_tweet = df_tweets[df_tweets["username"] == username].sort_values(by="id").set_index("id")
    # Map IDs to start from 0
    df_sorted_tweet = df_sorted_tweet.reset_index(drop=False)
    tweet_id_mapping = df_sorted_tweet["id"]
    tweet_node_features = df_sorted_tweet[[ #'user_id', 
                                          #  'full_text', 
                                          #  'user_mentions', 'source', 
                                          # 'in_reply_to_status_id', 'in_reply_to_user_id', 'quoted_status_id',
          'n_video_media', 'n_photo_media', 'retweet_count', 'favorite_count',
          'is_retweet']]
    
    # define edge_index
    user_map = user_id_mapping.reset_index().set_index("username").to_dict()
    tweet_map = tweet_id_mapping.reset_index().set_index("id").to_dict()

    post = df_sorted_tweet[["username", "id"]]
    post["username"] = post["username"].map(user_map["index"]).astype(int)
    post["tweet_id"] = post["id"].map(tweet_map["index"]).astype(int)

    post.drop("id", axis=1, inplace=True)

    post_edge_index = post[["tweet_id", "username"]].values.transpose()

    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_undirected

    data = HeteroData()
    data["user"].x = torch.from_numpy(user_node_features.to_numpy()).type(torch.float)
    data["user"].num_nodes = len(user_node_features)
    data["tweet"].x = torch.from_numpy(tweet_node_features.to_numpy()).type(torch.float)
    data["tweet"].num_nodes = len(tweet_node_features)
    data["tweet", "post", "user"].edge_index = torch.tensor(post_edge_index).type(torch.LongTensor)
    
    if with_label:
      data["y"] = torch.as_tensor(df_sorted_user["Y"]).type(torch.LongTensor)

    data = T.ToUndirected()(data)

    return data
  
  # generate graph
  graph_data = []
  for user in flattened_users:
    graph_data.append(get_graph(user["username"]))
  
  from torch_geometric.loader import DataLoader
  data_loader = DataLoader(graph_data, batch_size=10)

  return data_loader

def predict(model, data_loader):
    model.eval()
    all_preds = []
    
    for batch in data_loader:
        out = model(batch.x_dict, batch.edge_index_dict)
        all_preds.append(out["user"].argmax(dim=1))

    return all_preds

from twitter_scraper_selenium import get_profile_details

import json
def get_tweets_new_format(accounts, save_to_file=False, load_from_file=False, path=""):
  accounts = set(accounts)
  all_tweets = {}
  all_users = []
  import time
  for a in accounts:
    account_tweets = {}
    try:
      if load_from_file:
        with open(path+"/tw_"+a+".json", "r") as outfile:
          account_tweets = json.load(outfile)
        
        with open(path+"/ac_"+a+".json", "r") as outfile:
          user_detail = json.load(outfile)
      else:
        fname_tw = None
        fname_ac = None
        if save_to_file:
          fname_tw = path+"/tw_"+a
          fname_ac = path+"/ac_"+a
          account_tweets = json.load(scrape_profile(twitter_username=a,output_format="json",browser="firefox",tweets_count=30,filename=fname_tw))
          user_detail = json.loads(get_profile_details(twitter_username=a, filename=fname_ac))
        else:
          account_tweets = json.loads(scrape_profile(twitter_username=a,output_format="json",browser="firefox",tweets_count=30))
          user_detail = json.loads(get_profile_details(twitter_username=a))
      print(account_tweets)
      for i in account_tweets:
        account_tweets[i]["username_rt"] = account_tweets[i]["username"] if account_tweets[i]["is_retweet"] else ""
        account_tweets[i]["username"] = a

      all_tweets.update(account_tweets)
      all_users.append(user_detail)
    except Exception as e:
      import traceback
      traceback.print_exc()
      print(str(e))
  return all_users, all_tweets

# PATH = "model_new_format_v2.pt"
# model = torch.load(PATH)
# model.eval()

# raw_users, raw_tweets = get_tweets_new_format(["akhmad_mizkat"])
# processed_data = pre_process(raw_users, raw_tweets)
# print(len(raw_users))
# p = predict(model, processed_data)

# print(p[0].item())


def get_flattened_tweets(raw_tweets):
  flattened_tweets = []
  for _id in raw_tweets:
    r = raw_tweets[_id]
    flattened_tweets.append(dict(
        id=_id,
        username=r["username"],
        full_text=r["content"],
        user_mentions=r["mentions"],
        n_video_media=len(r["videos"]),
        n_photo_media=len(r["images"]),
        retweet_count=r["retweets"],
        favorite_count=r["likes"],
        replies=r["replies"],
        is_retweet = 1 if r["is_retweet"] else 0
    ))
  return pd.DataFrame(flattened_tweets)