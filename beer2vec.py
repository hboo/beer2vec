from gensim.models import word2vec
import click

class FileToSentence():
	def __init__(self, filename):
		self.filename = filename

	def __iter__(self):
		for line in open(self.filename, 'r'):
			ll = [i for i in unicode(line, 'utf-8').split()]
			yield ll

def initialize_model():
	sentences = FileToSentence('./sentences.txt')
	model = word2vec.Word2Vec(sentences=sentences, window=5, min_count=5, workers=4, hs=1)
	model.save('initial_w2v')

@click.command()
@click.option('--beer', prompt='Beer ')
def make_suggestion(beer):
	initialize_model()
	model = word2vec.Word2Vec.load('initial_w2v')
	results = model.wv.most_similar(positive=[beer])
	for suggestion in results:
		info = [x.replace('_', ' ').encode('utf-8') for x in suggestion[0].split(':')]
		print info[0] + ', ' + info[1] + ', ' + info[2]

if __name__ == '__main__':
	make_suggestion()
