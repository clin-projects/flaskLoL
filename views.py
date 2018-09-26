from flask import request
from flask import render_template
from flaskLoL import app
from flaskLoL.a_Model import ModelIt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flaskLoL.utils import flaskLoL_utils

# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
user = 'ccl' #add your Postgres username here      
host = 'localhost'
dbname = 'matches'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user, host = host) #add your Postgres password here

@app.route('/')
@app.route('/index')
def index():

	player_name = str(request.args.get('player_name'))

	if player_name == "None":
		player_name = "DoubleLift"
	
	matches = flaskLoL_utils.get_matches(player_name)

	return render_template("index.html", player_name = player_name,
		matches = matches)

@app.route('/lol_main')
def lolmain_page():
	gameID = int(request.args.get('gameID'))
	success = flaskLoL_utils.check_match_in_table(gameID, con)
	game_dat = flaskLoL_utils.read_match(gameID)
	plot_url = flaskLoL_utils.get_plot(gameID)
	video_html = flaskLoL_utils.test_video_html()
	winner, frames, predictions = flaskLoL_utils.read_and_predict(gameID, con)

	return render_template('lolmain.html', results = plot_url, game_dat= game_dat, winner=winner,
		video_html = video_html, frames = list(range(frames)), predictions = predictions,
		maxframe = min(frames, 40),
		gameID = gameID)