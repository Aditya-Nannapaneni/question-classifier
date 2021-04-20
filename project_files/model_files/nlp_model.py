import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, nltk
import pickle
import en_core_web_sm
nlp = en_core_web_sm.load()

import spacy
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import fbeta_score, accuracy_score
from scipy.sparse import hstack

with open('./model_files/model_coarse.bin', 'rb') as f_in:
        model_coarse = pickle.load(f_in)
        f_in.close()
    
with open('./model_files/model_fine.bin', 'rb') as f_in:
    model_fine = pickle.load(f_in)
    f_in.close()

with open('./model_files/le_coarse', 'rb') as f_in:
    le_coarse = pickle.load(f_in)
    f_in.close()

with open('./model_files/le_fine', 'rb') as f_in:
    le_fine = pickle.load(f_in)
    f_in.close()

with open('./model_files/count_vecs', 'rb') as f_in:
    count_vecs = pickle.load(f_in)
    f_in.close()

#functions
def text_clean(corpus, keep_list):
    '''
    AIM : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
    
    Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
            even after the cleaning process
    
    Output : Returns the cleaned text corpus
    
    '''
    cleaned_corpus = pd.Series()
    for row in corpus:
        qs = []
        for word in row.split():
            if word not in keep_list:
                p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                p1 = p1.lower()
                qs.append(p1)
            else : qs.append(word)
        cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
    return cleaned_corpus



def preprocess(corpus, keep_list, cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True):
    '''
    Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
    
    Input : 
    'corpus' - Text corpus on which pre-processing tasks will be performed
    'keep_list' - List of words to be retained during cleaning process
    'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                be performed or not
    'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                Stemmer. 'snowball' corresponds to Snowball Stemmer
    
    Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
    
    Output : Returns the processed text corpus
    
    '''
    if cleaning == True:
        corpus = text_clean(corpus, keep_list)
    
    if remove_stopwords == True:
        wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
        stop = set(stopwords.words('english'))
        for word in wh_words:
            stop.remove(word)
        corpus = [[x for x in x.split() if x not in stop] for x in corpus]
    else :
        corpus = [[x for x in x.split()] for x in corpus]
    
    if lemmatization == True:
        lem = WordNetLemmatizer()
        corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
    
    if stemming == True:
        if stem_type == 'snowball':
            stemmer = SnowballStemmer(language = 'english')
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
        else :
            stemmer = PorterStemmer()
            corpus = [[stemmer.stem(x) for x in x] for x in corpus]
    
    corpus = [' '.join(x) for x in corpus]
    

    return corpus

#nlp = spacy.load('en_core_web_sm')

def get_nlp_features(data:list):
    '''
    AIM: to featurize the data and extract ner, pos tags, lemmas, dependencies, orthographic features
    INPUT: data to featurize 
    OUTPUT: lists containing the specified features
    '''
    ners = []
    lemmas = []
    tags = []
    deps = []
    shapes = []
    
    for row in data:
        doc = nlp(row)
        ner = []
        lemma = []
        tag = []
        dep = []
        shape = []
        
        for token in doc:
            lemma.append(token.lemma_)
            tag.append(token.tag_)
            dep.append(token.dep_)
            shape.append(token.shape_)
        lemmas.append(" ".join(lemma))
        tags.append(" ".join(tag))
        deps.append(" ".join(dep))
        shapes.append(" ".join(shape))
        
        for ent in doc.ents:
            ner.append(ent.label_)
        ners.append(" ".join(ner))
    return [ners, lemmas, tags, deps, shapes]

def vectorize_features(count_vecs, features):
    '''
    AIM: transforms features into vectors by applying trained countvectorizers
    INPUT: list of countvectorizer and list of features
    OUTPUT: list of vectorized features
    '''
    vectorized_features = []
    for vec, feature in zip(count_vecs, features):
        vec_ft = vec.transform(feature)
        vectorized_features.append(vec_ft)
    return vectorized_features

def wrangle_features(features):
    #combining all features into 1 matrix and into compresses sparse row format
    #for easier computations
    
    return hstack(features).tocsr()
    

def predict(config):
    '''
    AIM: to predict the class label of the question
    INPUT: question to predict
    OUTPUT: required class label of the question to predict
    '''
    common_dot_words = ['U.S.', 'St.', 'Mr.', 'Mrs.', 'D.C.']
    #if type(config) == dict:
    question = config['text']
        
        
    text = [question]
    processed = preprocess(text, keep_list = common_dot_words, remove_stopwords = True)
    ques_features = get_nlp_features(processed)
    feature_vecs = vectorize_features(count_vecs, ques_features)
    feature_vecs = wrangle_features(feature_vecs)
    pred = [model_coarse.predict(feature_vecs), model_fine.predict(feature_vecs)]
    
    return [le_coarse.inverse_transform(pred[0])[0], le_fine.inverse_transform(pred[1])[0]]


