from flask import Flask

import os

app = Flask(__name__)
from flaskLoL import views

print(os.path.dirname(os.path.abspath(__file__)))