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

# @app.route('/artistlist')
# def artistlist():
# 	# example = {'song': "Yesterday, When I Was Mad", "artist": "Pet Shop Boys"}
# 	return render_template('artistlist.html')


@app.route('/search/<song>/<artist>', methods=['GET'])
def search(song, artist):
	data = df.loc[(df['song']==song.strip()) & (df['artist']==artist.strip())]
	data = data.to_json(orient='records')
	if len(data) == 0:
		data = df.loc[(df['song']==song)].to_json(orient='records')
	# print(data)
	return data

@app.route('/artist')
def getArtilist():
	artilist = set(list(df['artist']))
	# for art in artilist:
	# 	print(art)

	return '#'.join(artilist)

@app.route('/random', methods=['GET'])
def generateRandom():
	index = random.sample(range(len(df)), 1)
	data = df.ix[index].to_json(orient='records', lines=True)
	# data = json.loads(data)
	# print(data)
	return data

@app.route('/runCNN/<song>/<artist>', methods=['GET'])
def runCNN(song, artist):
	data = df.loc[(df['song']==song.strip()) & (df['artist']==artist.strip())]
	lyric = data['text']

	# To runn CNN model and return the probability
	return None

@app.route('/model', methods=['GET'])
def modelPage():
	#Todo, have to write a page to introduce our model
	return render_template('model.html')