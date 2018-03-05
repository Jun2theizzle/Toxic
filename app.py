import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from sklearn.metrics import mean_squared_error, r2_score
import pickle 

tiny = False
df = None
test = None

if(tiny == True):
    df=pd.read_csv('train_tiny.csv')
else:
    df = pd.read_csv('train.csv')

word_vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word', stop_words='english')
comment_text_bigram = word_vectorizer.fit_transform(df['comment_text'])

print 'Vectorize done'

# print word_vectorizer.vocabulary_

# toxic_label = df['toxic'].values
# # Create linear regression object
# clf = SVC()

# # Train the model using the training sets
# clf.fit(comment_text_bigram, toxic_label)

# print 'Training done'

# pickle.dump(clf, open('bigramModel.sav', 'wb'))

if(tiny == True):
    test = pd.read_csv('test_tiny.csv')
else:
    test = pd.read_csv('test.csv')

test_bigram = word_vectorizer.transform(test['comment_text'].values)

loaded_model = pickle.load(open('bigramModel.sav', 'rb'))
toxicity_predict = loaded_model.predict(test_bigram)

for i in range(len(toxicity_predict)):
    print '{}, {}'.format(test['id'][i], toxicity_predict[i])