import json
from sklearn.model_selection import train_test_split

FILE = './beeradvocate_reviews.txt'

count = 0
with open('./training_sentences_20180121.txt', 'w+') as train:
	with open('./testing_sentences_20180121.txt', 'w+') as test:
		with open(FILE, 'r') as f:
			data = json.load(f)
			for x in data:
				user_data = []
				for beer in x['reviews']:
					if beer['rating'] >= 4.0:
						entity = beer['name'].encode('utf-8') + ':' + beer['brewery'].encode('utf-8') + ':' + beer['style'].encode('utf-8')
						user_data.append(entity.replace(' ', '_'))

				sentences_train, sentences_test = train_test_split(user_data, test_size=10)
				for x in sentences_train:
					train.write(x + ' ')
				train.write('\n')

				for x in sentences_test:
					test.write(x + ' ')
				test.write('\n')