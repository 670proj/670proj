from flask_wtf import FlaskForm
from wtforms import StringField,  SubmitField
from wtforms.validators import DataRequired

class SearchForm(FlaskForm):
    songname = StringField('Song Name', validators=[DataRequired()])
    artist = StringField('Artist', validators=[DataRequired()])
    submit = SubmitField('Search')