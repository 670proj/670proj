from pymongo import MongoClient
import pandas as pd
import json
try:
    conn = MongoClient()
    print("Connected successfully!!!")
except:  
    print("Could not connect to MongoDB")
 

df = pd.read_csv("songcleaned.csv")
# print(df['song'])
print(len(set(df['artist'])))
# records = json.loads(df.T.to_json()).values()
# # database
# db = conn.cs670
 
# # # Created or Switched to collection names: my_gfg_collection
# collection=db.songcat
# collection.insert(records)

