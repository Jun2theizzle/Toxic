import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error, r2_score
import pickle 
import json

tiny = False
df = None
test = None
vocabulary = None

if(tiny == True):
    df=pd.read_csv('train_tiny.csv')
else:
    df = pd.read_csv('train.csv')

with open('unigram.json', 'r') as f:
    vocabulary = json.load(f)

word_vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer='word', stop_words='english', vocabulary=vocabulary)
comment_text_bigram = word_vectorizer.fit_transform(df['comment_text'])

print 'Vectorize done'
print len(word_vectorizer.vocabulary_)
# with open('unigram.json', 'w') as vocab:
#     vocab.write(json.dumps(word_vectorizer.vocabulary_))

labels = df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
# Create linear regression object
# clf = SVC()

# # Train the model using the training sets
# clf.fit(comment_text_bigram, labels)


neigh = KNeighborsClassifier(n_neighbors=6)

neigh.fit(comment_text_bigram, labels) 
print 'Training done'

# pickle.dump(neigh, open('knnModel.sav', 'wb'))

if(tiny == True):
    test = pd.read_csv('test_tiny.csv')
else:
    test = pd.read_csv('test.csv')

test_bigram = word_vectorizer.transform(test['comment_text'].values)

# loaded_model = pickle.load(open('knnModel.sav', 'rb'))
with open('results.txt', 'w') as f:
    f.write('id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n')
    for i in range(test_bigram.getnnz()):
        toxicity_predict = neigh.predict(test_bigram[i])
        prediction = ','.join(str(pred) for pred in toxicity_predict[0])
        f.write('{},{}\n'.format(test['id'][i], prediction))