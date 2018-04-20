from flask import Flask
from config import Config
# from pymongo import MongoClient

app = Flask(__name__)
# mongo = MongoClient().cs670
app.config.from_object(Config)

from app import routes