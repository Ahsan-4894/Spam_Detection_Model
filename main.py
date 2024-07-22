import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer
from gensim.utils import tokenize

#.csv file https://github.com/Buzzpy/Python-Projects/blob/main/Spam-detection/spam.csv

spam = []
non_spam = []
df = pd.read_csv("spam.csv")
spam = df[df["v1"]=="spam"]["v2"].tolist()
non = df[df["v1"]=="ham"]["v2"].tolist()

# #test sentence for model
spam_test = ["FreeMsg collect the prize reward!"
            ]

def tokenize_sentence(sentence):
    porter = PorterStemmer()
    removed_stop_sentence = remove_stopwords(sentence)
    porter = porter.stem(removed_stop_sentence)
    sentence_tokens = tokenize(porter)
    return list(sentence_tokens)

spams_tokenized=[]
non_spams_tokenized=[]
dictionary = set()


for sentence in spam:
    sentence_token = tokenize_sentence(sentence)
    spams_tokenized.append(sentence_token)    
    dictionary = dictionary.union(sentence_token)

for sentence in non:
    sentence_token = tokenize_sentence(sentence)
    non_spams_tokenized.append(sentence_token)
    dictionary = dictionary.union(sentence_token)

total_words = len(dictionary)
total_spam_messages = len(spams_tokenized)
total_messages = total_spam_messages + len(non_spams_tokenized)

#P(p|spam)
p_spam = total_spam_messages / total_messages

def total_word_in_messages(word, messages):
    count=0
    for w in messages:
        if word in w:
            count = count+1
 
    return count

for test_sentence in spam_test:
    final_prob=1
    test_sentence = tokenize_sentence(test_sentence)

    for word in test_sentence:
        #P(w/spam)
        word_count = total_word_in_messages(word, spams_tokenized)
        p_w_spam = (word_count/total_spam_messages)

        #P(w)
        word_count = total_word_in_messages(word, spams_tokenized) + total_word_in_messages(word, non_spams_tokenized)
        p_w = (word_count/total_messages)

        #P(spam/w)
        if(p_w==0):
            p_w=1
        p_spam_w = ((p_w_spam*p_spam)/p_w)
        print('-----------------')
        print("For Word:", word)
        print("p(spam)", p_spam)
        print("p(w)", p_w)
        print("p(w|spam)", p_w_spam)
        print("p(spam|w)",p_spam_w)
        print('-------------------')
        final_prob *= p_spam_w
    
    #p(spam|words)
    print("Probablity of sentence(in tokenized form)", test_sentence,"is:",final_prob)


