from gensim.models import word2vec
import click
from itertools import izip
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class FileToSentence():
	def __init__(self, filename):
		self.filename = filename

	def __iter__(self):
		for line in open(self.filename, 'r'):
			ll = [i for i in unicode(line, 'utf-8').split()]
			yield ll

def initialize_model(win, ct, sz):
	sentences = FileToSentence('./training_sentences_20180121.txt')
	model = word2vec.Word2Vec(sentences=sentences, size=sz, window=win, min_count=ct, workers=4, hs=1)
	model.save('model_to_viz_02_25')
	return model

#model = initialize_model(1, 1, 50)

model = word2vec.Word2Vec.load('model_to_viz_02_25')
num = 100

count = 0
labels = []
tokens = []
fpath = 'fig_' + str(num) + 'beer1n byu67 m'
for word in model.wv.vocab:
	count += 1
	if count % 3:
		pass
	else:
		tokens.append(model[word])
		if count % num:
			labels.append('')
		else:
			try:
				labels.append(word.split(':')[0].replace('_', ' '))
			except:
				labels.append('')

# Sample labels
labels = [label if indx % 600 else '' for indx,label in enumerate(labels)]


tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(tokens)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
    
plt.figure(figsize=(16, 16)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.savefig(fpath)

