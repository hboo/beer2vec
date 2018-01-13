import json

FILE = './beeradvocate_reviews.txt'

with open('./sentences.txt', 'w+') as sentences:
	with open(FILE, 'r') as f:
		data = json.load(f)
		for x in data:
			for beer in x['reviews']:
				if beer['rating'] >= 4.0:
					entity = beer['name'].encode('utf-8') + ':' + beer['brewery'].encode('utf-8') + ':' + beer['style'].encode('utf-8')
					sentences.write(entity.replace(' ', '_') + ' ')
			sentences.write('\n')