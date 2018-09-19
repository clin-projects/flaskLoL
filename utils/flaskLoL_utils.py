from io import BytesIO
import base64
import os 
import json
import datetime
from pytz import timezone
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def read_match(gameID):
	match_file = '/Users/ccl/anaconda3/envs/LoL/flaskapp/flaskLoL/dat/match_%d.json' % (gameID)
	timeline_file = match_file.replace('match','timeline')
	match = json.load(open(match_file, 'rb'))
	timeline = json.load(open(timeline_file, 'rb'))

	player_names = [match['participantIdentities'][i]['player']['summonerName'] for i in range(10)]
	champion_ids = [match['participants'][i]['championId'] for i in range(10)]


	base_photo = './static/img/'

	champion_photos = [base_photo + str(x) + '.png' for x in champion_ids]

	ms = match['gameCreation']
	t = datetime.datetime.fromtimestamp(ms/1000.0)
	t = t.astimezone(timezone('US/Pacific'))
	game_start = t.strftime('%B %d, %Y: %H:%M PST')

	duration = match['gameDuration'] / 60
	duration_min = int(duration)
	duration_sec = round((duration *60) % 60)
	game_duration = '%d minutes, %d seconds' % (duration_min, duration_sec)

	game_dat = dict(names = player_names, champions = champion_ids, photos = champion_photos,
		game_start = game_start, duration = game_duration)

	dir_path = os.path.dirname(os.path.realpath(__file__))

	return game_dat

def get_plot(gameID):

	# first get relevant pieces of game data
	# preprocess (scale values and get into right form)
	# then get model coefficients and calculate probability over time
	# then generate plot
	# should also show victory / defeated

	#https://stackoverflow.com/questions/41459657/how-to-create-dynamic-plots-to-display-on-flask
	img = BytesIO()

	y = [1,2,3,4,5]
	x = [0,2,1,3,4]
	plt.plot(x,y)
	plt.savefig(img, format='png')
	img.seek(0)

	plot_url = base64.b64encode(img.getvalue()).decode()
	return plot_url

def test_video_html():

	#=========================================
	# Create Fake Images using Numpy 
	# You don't need this in your code as you have your own imageList.
	# This is used as an example.

	imageList = []
	n = 40
	x = np.array(range(n))
	y = np.random.random(n)
	imageList = np.array([[x[i],y[i]] for i in range(n)])

	#=========================================
	# Animate Fake Images (in Jupyter)

	def getImageFromList(x):
	    
	    return imageList[x]

	fig = plt.figure(figsize=(10, 10))
	ims = []
	for i in range(len(imageList)):
	#for i in range(1):
	    im, = plt.plot(imageList[:i+1][:,0],imageList[:i+1][:,1], animated=True,color='blue',marker='o')
	    #im = plt.scatter(imageList[i][0],imageList[i][1], animated=True,color='blue')
	    #im.plot(imageList[:i+1][:,0],imageList[:i+1][:,1], animated=True,color='blue')
	    ims.append([im])

	delay = 1000
	ani = animation.ArtistAnimation(fig, ims, interval=delay, blit=True, repeat_delay=1000)
	plt.close()

	# Show the animation
	html = ani.to_html5_video()

	html = html.replace('<video width="1000" height="1000"', 
				 '<video width="400" height="400"')

	return HTML(html)