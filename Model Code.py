import pandas as pd
import nltk
import re
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('balanced_reviews.csv')

def data_cleaning():
    global features
    global labels
    df = pd.read_csv('balanced_reviews.csv')
    df.dropna(inplace = True)
    df = df[df['overall'] != 3]
    df['Positivity'] = np.where(df['overall'] > 3, 1, 0 )
    df.to_csv('balanced_reviews.csv', index = False)  
        
    corpus = []
    
    for i in range(0, 527386):
    
        review = re.sub('[^a-zA-Z]', ' ', df.iloc[i, 1])
        review = review.lower()
        review = review.split()
        review = [word for word in review if not word in stopwords.words('english')]
        ps =  PorterStemmer()
        review = [ps.stem(word) for word in review]
        review = " ".join(review)
        corpus.append(review)
        
    features = corpus
    labels = df.iloc[:,-1]

def model_build():
    global model
    global vect
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = 42 ) 
    vect = TfidfVectorizer(min_df = 5).fit(features_train)
    features_train_vectorized = vect.transform(features_train)
    model = LogisticRegression()
    model.fit(features_train_vectorized, labels_train)

def model_vocab_dump():
    pickle.dump(vect.vocabulary_, open('feature.pkl','wb'))
    with open('pickle_model.pkl', 'wb') as file:
        pickle.dump(model, file)

def main():
    global features
    global labels
    global model
    global vect
    print("Creating data sampling...")
 #   data_sample()
    print("Cleaning data...")
    data_cleaning()
    print("Building Model...")
    model_build()
    print("Dumping model in a pickle file...")
    model_vocab_dump()
    print("Pickle file is ready to be used...")
    features = None
    labels = None
    model = None
    vect = None
	
if __name__ == '__main__':
    main()
