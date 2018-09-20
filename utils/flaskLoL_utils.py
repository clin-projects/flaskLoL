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

def calculate_team_total_and_difference(dat, frames):
	team_total = np.array([np.sum(dat[:,:5],axis=1),np.sum(dat[:,5:],axis=1)]).transpose()
	team_total_diff = np.concatenate((team_total,(team_total[:,0] - team_total[:,1]).reshape((frames,1))),axis=1)
	return team_total_diff

def get_kills_by_match(match):
	"""
	needs timeline file, and outputs kills by frame as [team_0 kills, team_1 kills, difference]
	"""
	num_frames = len(match['frames'])
	
	kills_by_frame = []
	
	for i in range(num_frames):
		num_events = len(match['frames'][i]['events'])
		team_0_kills = 0
		team_1_kills = 0
		for j in range(num_events):
			cur_event = match['frames'][i]['events'][j]
			cur_event_type = cur_event['type']
			if cur_event_type == 'CHAMPION_KILL':
				if cur_event['killerId'] <= 5:
					team_0_kills += 1 
				else:
					team_1_kills += 1
				#print(cur_killer_team)
		kills_by_frame.append([team_0_kills, team_1_kills, team_0_kills - team_1_kills])
		
	return kills_by_frame

def get_buildings_by_match(match):
	"""
	gets counts of buildings destroyed by frame
	"""
	
	# These are ranked from furthest-to-closest from the Nexus
	map_buildings = {
		'OUTER_TURRET'    : 0,
		'INNER_TURRET'    : 1,
		'BASE_TURRET'     : 2,
		'UNDEFINED_TURRET': 3,
		'NEXUS_TURRET'    : 4
	}
	
	building_counts = []
	
	num_frames = len(match['frames'])
	
	buildings_counts_by_frame = []
	
	for i in range(num_frames):
		num_events = len(match['frames'][i]['events'])
		team_0_kills = 0
		team_1_kills = 0
		cur_building_counts = np.zeros((3,5))
		for j in range(num_events):
			cur_event = match['frames'][i]['events'][j]
			cur_event_type = cur_event['type']
			if cur_event_type == 'BUILDING_KILL':
				cur_building_type = cur_event['buildingType']
				cur_tower_type = cur_event['towerType']
				cur_killer_team = 0 if cur_event['killerId'] <= 5 else 1
				cur_building_counts[cur_killer_team,map_buildings[cur_tower_type]] += 1
		cur_building_counts[2,:] = cur_building_counts[0,:] - cur_building_counts[1,:]
		buildings_counts_by_frame.append(cur_building_counts)
	return np.array(buildings_counts_by_frame)

def get_monsters_by_match(match):
	"""
	gets counts of monsters killed by frame
	"""
	
	map_monsters = {
		'BARON_NASHOR'    : 0,
		'RIFTHERALD'      : 1,
		'AIR_DRAGON'      : 2,
		'EARTH_DRAGON'    : 3,
		'WATER_DRAGON'    : 4,
		'FIRE_DRAGON'     : 5,
		'ELDER_DRAGON'    : 6
	}
	
	monster_counts = []
	
	num_frames = len(match['frames'])
	
	monster_counts_by_frame = []
	
	for i in range(num_frames):
		num_events = len(match['frames'][i]['events'])
		cur_monster_counts = np.zeros((3,7))
		for j in range(num_events):
			cur_event = match['frames'][i]['events'][j]
			cur_event_type = cur_event['type']
			if cur_event_type == 'ELITE_MONSTER_KILL':
				cur_monster_type = cur_event['monsterType']
				cur_killer_team = 0 if cur_event['killerId'] <= 5 else 1
				if cur_monster_type == 'DRAGON':
					cur_monster_type = cur_event['monsterSubType']
				cur_monster_counts[cur_killer_team, map_monsters[cur_monster_type]] += 1
		cur_monster_counts[2,:] = cur_monster_counts[0,:] - cur_monster_counts[1,:]
		monster_counts_by_frame.append(cur_monster_counts)
	return np.array(monster_counts_by_frame)

def get_gold(match):

	total_gold = []    
	
	frames = len(match['frames'])
	for frame in range(frames):
		frame_total_gold = [match['frames'][frame]['participantFrames'][str(i)]['totalGold'] for i in range(1,11)]
		total_gold.append(frame_total_gold)
	
	total_gold = calculate_team_total_and_difference(np.array(total_gold), frames)

	return total_gold

def get_parameters(match_file, timeline_file, gold_min, gold_max):
	
	match = json.load(open(match_file, 'rb'))
	timeline = json.load(open(timeline_file, 'rb'))
	
	gold = get_gold(timeline)[:,2]
	gold_rescaled = (gold - gold_min) / (gold_max - gold_min) * (2) - 1
	
	kills = get_kills_by_match(timeline)
	kills = np.cumsum(np.array(kills)[:,2])
	
	monsters = get_monsters_by_match(timeline)
	monsters = np.cumsum(np.array(monsters)[:,2,:],axis=1)
	
	buildings = get_buildings_by_match(timeline)
	buildings = np.cumsum(np.array(buildings)[:,2,:],axis=1)
		
	winner = (match['teams'][0]['win'] == 'Win')*1
	
	return gold_rescaled, kills, monsters, buildings, winner

def generate_X_one(gold, kills, monsters, buildings, start_frame, end_frame):
	current_gold = gold[start_frame: end_frame+1]
	current_kills = kills[start_frame: end_frame+1]

	current_monsters = monsters[start_frame: end_frame+1]
	current_monsters = current_monsters.reshape(current_monsters.shape[0] * current_monsters.shape[1])
	
	current_buildings = buildings[start_frame: end_frame+1]
	current_buildings = current_buildings.reshape(current_buildings.shape[0] * current_buildings.shape[1])
	
	return np.concatenate((current_gold, current_kills, current_monsters, current_buildings))

def predict(lag, coef, gold_rescaled, kills, monsters, buildings):
	y_predictions = []

	maxframe = min(len(gold_rescaled), 40)
	for frame in range(maxframe):
		start_frame = max(0,frame-lag)
		X = generate_X_one(gold_rescaled, kills, monsters, buildings, start_frame, frame)
		cur_coef = coef[frame]
		y_pred = 1 / (1 + np.exp(-1 * np.dot(X, cur_coef)))
		y_predictions.append(y_pred)
	return y_predictions

def read_and_predict(gameID):
	param_file = '/Users/ccl/anaconda3/envs/LoL/flaskapp/flaskLoL/static/param/model_parameters.npz'
	params = np.load(param_file)
	gold_min = params['gold_min']
	gold_max = params['gold_max']
	coef = params['coef']

	match_file = '/Users/ccl/anaconda3/envs/LoL/flaskapp/flaskLoL/dat/match_%d.json' % (gameID)
	timeline_file = match_file.replace('match','timeline')

	gold_rescaled, kills, monsters, buildings, winner = get_parameters(match_file, timeline_file, gold_min, gold_max)

	lag = 5

	y_predictions = predict(lag, coef, gold_rescaled, kills, monsters, buildings)

	return winner, list(range(len(gold_rescaled))), y_predictions




