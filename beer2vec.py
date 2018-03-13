from gensim.models import word2vec
import click
from itertools import izip
import numpy as np

class FileToSentence():
	# generator which constructs sentences from a file of json words
	def __init__(self, filename):
		self.filename = filename

	def __iter__(self):
		for line in open(self.filename, 'r'):
			ll = [i for i in unicode(line, 'utf-8').split()]
			yield ll

def initialize_model(win=1000, ct=1, sz=100):
	#build sentences from training data
	sentences = FileToSentence('./training_sentences_20180121.txt')
	#initialize and train model
	model = word2vec.Word2Vec(sentences=sentences, size=sz, window=win, min_count=ct, workers=4, hs=1)
	model.save('initial_w2v_20180121')
	return model

@click.command()
@click.option('--beer', prompt='Beer ')
def make_suggestion(beer):
	# load model from disk
	model = word2vec.Word2Vec.load('initial_w2v')
	# this method returns the beers with the highest cosine similarities
	results = model.wv.most_similar(beer)
	for suggestion in results:
		info = [x.replace('_', ' ').encode('utf-8') for x in suggestion[0].split(':')]
		print info[0] + ', ' + info[1] + ', ' + info[2]

def evaluate():
	model = word2vec.Word2Vec.load('initial_w2v_20180121')
	x = 0
	y = 0
	matches = []
	with open('testing_sentences_20180121.txt', 'r') as test_data, open('training_sentences_20180121.txt', 'r') as train_data:
		for test_line, train_line in izip(test_data, train_data):
			match_count = 0
			
			def get_results(data):
				try:
					# predict
					return model.wv.most_similar(positive=data)
				except KeyError as e:
					# the beer may not exist in the model because it was not commonly reviewed
					not_found = e.message[6:-19]
					if len(data) < 2:
						# no recommendation was possible - no beers found in model
						return ''
					# unless the list is empty, remove the unrepresented beer and 
					# try the prediction again with the remaining beers
					data.remove(not_found)
					get_results(data)

			results = get_results(test_line.split(' ')[:-1])

			if results:
				# count how many recommendations were represented in the same user's history
				x +=1
				for suggestion in results:
					if suggestion[0].encode('utf-8') in train_line:
						match_count += 1
				matches.append(match_count)
			else:
				y += 1

		return matches

if __name__ == '__main__':
	make_suggestion()
	initialize_model()
	evaluate()
	
	# use grid search on these three params
	windows = [1000, 500, 100, 50, 10, 1]
	min_count = [1, 2, 3, 4, 5]
	size = [5, 10, 50, 100]
	with open('results_20180205.txt', 'w+') as res:
		for win in windows:
			for ct in min_count:
				for sz in size:
					initialize_model(win, ct, sz)
					results = evaluate()
					l = len(results)
					pos = len([x for x in results if x != 0])
					neg = l - pos
					res.write('{}, {}, {}, {}, {}, {}, {}\n'.format(win, ct, sz, l, pos, neg, np.mean(results)))
