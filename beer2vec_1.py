from gensim.models import word2vec
import click
from itertools import izip

class FileToSentence():
	def __init__(self, filename):
		self.filename = filename

	def __iter__(self):
		for line in open(self.filename, 'r'):
			ll = [i for i in unicode(line, 'utf-8').split()]
			yield ll


def parse_beers():
	import json
	from sklearn.model_selection import train_test_split

	FILE = './beeradvocate_reviews.txt'

	count = 0
	with open('./training_sentences_20180204.txt', 'w+') as train_data:
		with open('./verify_sentences_20180204.txt', 'w+') as verify:
			with open('./testing_sentences_20180204.txt', 'w+') as test:

				with open(FILE, 'r') as f:
					data = json.load(f)
					for x in data:
			   			count += 1

						user_data = []
						test_data = []
						for beer in x['reviews']:
							if beer['rating'] >= 4.0:
								entity = beer['name'].encode('utf-8') + ':' + beer['brewery'].encode('utf-8') + ':' + beer['style'].encode('utf-8')
								if count < 235:
									test_data.append(entity.replace(' ', '_'))
								else:
									user_data.append(entity.replace(' ', '_'))

						sentences_verify, sentences_test = train_test_split(test_data, test_size=0.1)
						#import pdb;pdb.set_trace()
						for x in sentences_verify:
							verify.write(x + ' ')
						verify.write('\n')

						for x in sentences_test:
							test.write(x + ' ')
						test.write('\n')

						for x in user_data:
							train_data.write(x + ' ')
						train_data.write('\n')


def initialize_model():
	sentences = FileToSentence('./training_sentences_20180204.txt')
	model = word2vec.Word2Vec(sentences=sentences, window=100, min_count=1, workers=4, hs=1)
	model.save('initial_w2v_20180204')
	return model

@click.command()
@click.option('--beer', prompt='Beer ')
def make_suggestion(beer):
	model = word2vec.Word2Vec.load('initial_w2v')
	results = model.wv.most_similar(beer)
	for suggestion in results:
		info = [x.replace('_', ' ').encode('utf-8') for x in suggestion[0].split(':')]
		print info[0] + ', ' + info[1] + ', ' + info[2]

def evaluate():
	model = word2vec.Word2Vec.load('initial_w2v_20180204')
	x = 0
	y= 0
	with open('testing_sentences_20180204.txt', 'r') as test_data, open('verify_sentences_20180204.txt', 'r') as verify_data:
		for test_line, verify_line in izip(test_data, verify_data):
			match_count = 0
			
			def get_results(data):
				try:
					return model.wv.most_similar(positive=data)
				except KeyError as e:
					not_found = e.message[6:-19]
					if len(data) < 2:
						return 'x'
					data.remove(not_found)
					get_results(data)
				except Exception as e:
					pass

			results = get_results(test_line.split(' ')[:-1])
			#print results

			if results:
				x +=1
				for suggestion in results:
					if suggestion[0].encode('utf-8') in verify_line:
						match_count += 1
				print match_count
			else:
				y += 1

		print 'not found:{}'.format(str(y))
		print 'found with 2:{}'.format(str(x))



if __name__ == '__main__':
	#make_suggestion()
	parse_beers()
	initialize_model()
	evaluate()