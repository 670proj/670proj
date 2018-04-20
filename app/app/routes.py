from app import app
from flask import render_template, flash, redirect, url_for
from app.model.forms import SearchForm
import json
import random
import pandas as pd

df = pd.read_csv("songcleaned.csv")

@app.route('/')
@app.route('/index')
def index():
	index = random.sample(range(len(df)), 1)
	data = df.ix[index].to_json(orient='records', lines=True)
	data = json.loads(data)
	# example = {'song': "Yesterday, When I Was Mad", "artist": "Pet Shop Boys"}
	return render_template('index.html', example=data)

@app.route('/search/<song>/<artist>', methods=['GET'])
def search(song, artist):
	data = df.loc[(df['song']==song.strip()) & (df['artist']==artist.strip())]
	data = data.to_json(orient='records')
	if len(data) == 0:
		data = df.loc[(df['song']==song)].to_json(orient='records')
	# print(data)
	return data


@app.route('/random', methods=['GET'])
def generateRandom():
	index = random.sample(range(len(df)), 1)
	data = df.ix[index].to_json(orient='records', lines=True)
	# data = json.loads(data)
	# print(data)
	return data
