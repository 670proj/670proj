from app import app
from flask import render_template, flash, redirect, url_for
from app.model.forms import SearchForm
# from app import mongo
import pandas as pd

df = pd.read_csv("songcleaned.csv");

@app.route('/')
@app.route('/index')
def index():
	example = {'song': "Yesterday, When I Was Mad", "artist": "Pet Shop Boys"}
	return render_template('index.html', example=example)

@app.route('/search/<song>/<artist>', methods=['GET'])
def search(song, artist):
	
	data = df.loc[(df['song']==song) & (df['artist']==artist)]
	return data.to_json(orient='records')




