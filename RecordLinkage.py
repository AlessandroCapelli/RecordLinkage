import pandas
import recordlinkage
from recordlinkage.preprocessing import clean
from recordlinkage.index import SortedNeighbourhood
from recordlinkage.index import Random
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

##### CARICAMENTO DATASET

dataset = pandas.read_csv('dataset.txt', skip_blank_lines=True, sep='\t', names=['source', 'isbn', 'title', 'author_list'], encoding='utf8')

##### PREPROCESSING: PULIZIA DATASET

dataset['source'] = clean(dataset['source'])
dataset['isbn'] = clean(dataset['isbn'])
dataset['title'] = clean(dataset['title'])
dataset['author_list'] = clean(dataset['author_list'])

# dataset.to_csv('dataset.csv', sep=';', index=True)

##### RECORD LINKAGE: DEDUPLICATION

# Calcolo benchmark per avere un riferimento di correttezza
indexer_benchmark = recordlinkage.Index()
indexer_benchmark.block('isbn')
benchmark = indexer_benchmark.index(dataset)

#### Threshold-based methods
print("#### Threshold-based methods")

indexer = recordlinkage.Index()
indexer.add(SortedNeighbourhood('title', window=31)) # con window=1 si ha blocking
indexer.add(Random((len(benchmark) * 2)))

candidate_links = indexer.index(dataset)
compare = recordlinkage.Compare()
print("Candidate links: {}".format(len(candidate_links)))
# print(candidate_links)

compare.string('title', 'title', threshold=0.6, method='levenshtein', label='compare_title')
compare.string('author_list', 'author_list', threshold=0.4, method='levenshtein', label='compare_author_list')

features = compare.compute(candidate_links, dataset)
#print(features)
d = 0.9 # Rappresenta la distanza per decidere la soglia di match/not match/possible match
predictions = features[features.sum(axis=1) > d]
#print(predictions)
print("Matches: {}".format(len(predictions)))

#### PERFORMANCE

confusion_matrix = recordlinkage.confusion_matrix(benchmark, predictions, len(features))

print("Precision: {}".format(recordlinkage.precision(confusion_matrix)))
print("Recall: {}".format(recordlinkage.recall(confusion_matrix)))
print("F-Measure: {}".format(recordlinkage.fscore(confusion_matrix)))

#### Supervised learning methods
print("#### Supervised learning methods")

train, test = train_test_split(features, test_size=0.3)

train_matches_index = train.index & benchmark
test_matches_index = test.index & benchmark

naiveBayesClassifier = recordlinkage.NaiveBayesClassifier(binarize=0.5)
naiveBayesClassifier.fit(train, train_matches_index)

predictions = naiveBayesClassifier.predict(test)

#### PERFORMANCE

confusion_matrix = recordlinkage.confusion_matrix(test_matches_index, predictions, len(test))

print("Precision: {}".format(recordlinkage.precision(confusion_matrix)))
print("Recall: {}".format(recordlinkage.recall(confusion_matrix)))
print("F-Measure: {}".format(recordlinkage.fscore(confusion_matrix)))

#### Unsupervised learning methods
print("#### Unsupervised learning methods")

train, test = train_test_split(features, test_size=0.3)

test_matches_index = test.index & benchmark

kmeans = recordlinkage.KMeansClassifier()
result_kmeans = kmeans.fit_predict(train)

predictions = kmeans.predict(test)

#### PERFORMANCE

confusion_matrix = recordlinkage.confusion_matrix(test_matches_index, predictions, len(test))

print("Precision: {}".format(recordlinkage.precision(confusion_matrix)))
print("Recall: {}".format(recordlinkage.recall(confusion_matrix)))
print("F-Measure: {}".format(recordlinkage.fscore(confusion_matrix)))
