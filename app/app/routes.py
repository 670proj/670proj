from app import app
from flask import render_template, flash, redirect, url_for
from bson.json_util import dumps
from app.model.forms import SearchForm
from app import mongo


@app.route('/')
@app.route('/index')
def index():
	example = {'song': "Yesterday, When I Was Mad", "artist": "Pet Shop Boys"}
	return render_template('index.html', example=example)

@app.route('/search/<song>/<artist>', methods=['GET'])
def search(song, artist):
	# {"song":song, "artist":artist}
	items = mongo.songcat.find({'song':song, 'artist':artist})
	# df.
	data = dumps(items)
	print(data)
	print(type(data))
	return data




