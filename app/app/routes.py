from app import app
from flask import render_template, flash, redirect, url_for, request
from app.model.forms import SearchForm
from app.model.Model import Model
import json
import random
import pandas as pd
import html

df = pd.read_csv("songcleaned2.csv")
model = Model('data/checkpoints/model-3200', 'data/word2vector3.model');

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

# @app.route('/runCNN/<song>/<artist>', methods=['GET'])
# def runCNN(song, artist):
# 	data = df.loc[(df['song']==song.strip()) & (df['artist']==artist.strip())]
# 	lyric = data['text']
# 	cat_pred = model.prediction(list(lyric)[0])
# 	return cat_pred

@app.route('/runCNN', methods=['GET'])
def runCNN():
	song=html.unescape(request.args['song'])
	artist=html.unescape(request.args['artist'])

	data = df.loc[(df['song']==song.strip()) & (df['artist']==artist.strip())]
	lyric = data['text']
	cat_pred = model.prediction(list(lyric)[0])
	return cat_pred

@app.route('/model', methods=['GET'])
def modelPage():
	#Todo, have to write a page to introduce our model
	return render_template('model.html')

@app.route('/result', methods=['GET'])
def resultPage():
	#Todo, have to write a page to introduce our model
	song=request.args['song']
	artist=request.args['artist']
	data = df.loc[(df['song']==song.strip()) & (df['artist']==artist.strip())]
	# data = data.to_json(orient='records')
	# if len(data) == 0:
	# 	data = df.loc[(df['song']==song)].to_json(orient='records')

	if data.empty:
		return render_template('result.html', keys=None, data=None)

	data = data.to_dict('records')[0]
	keys = ['Song Name', 'Artist Name', 'Lyric', 'Cat']
	values = {keys[0]: data['song'],  keys[1]: data['artist'], keys[2]:data['text'], keys[3]: data['cat']}
	return render_template('result.html', keys=keys, data=values)