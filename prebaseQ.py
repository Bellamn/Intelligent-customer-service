#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:22:48 2018

@author: dyh
"""

from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pandas as pd
from gensim import corpora, models, similarities

tr4w = TextRank4Keyword()
data = pd.read_excel("/home/dyh/Desktop/zhineng/FAQ.xlsx")
text = "?".join(data.QUESTION)
tr4w.analyze(text=text, lower=True, window=2)
sentence =  tr4w.words_no_stop_words 

      
dictionary = corpora.Dictionary(sentence)
dictionary.save("/home/dyh/Desktop/zhineng/deerwester.dict")
corpus = [dictionary.doc2bow(text) for text in sentence]
#corpora.MmCorpus.serialize('/home/dyh/Desktop/zhineng/deerwester.mm', corpus)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)
#lsi.save('/home/dyh/Desktop/zhineng/model.lsi')

index = similarities.MatrixSimilarity(lsi[corpus])

def preAns(new_doc):
    tr4w.analyze(text= new_doc, lower=True, window=2)
    new_doc =  tr4w.words_no_stop_words 
    new_vec = dictionary.doc2bow(new_doc[0])
    new_lsi = lsi[new_vec]
    sims = index[new_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims
        
active = True
while active:
    str = input()
    if str == 'quit':
        active = False
    else : 
		sims = preAns(str)
		if sims[1][1] < 0.7:
			print("请详细描述您的问题")
		else : 
			print(data.QUESTION[sims[0][0]])
			print(data.ANSWER[sims[0][0]])
			print(sims[0:3])
			
test = pd.read_excel(r"C:\Users\dyh\Desktop\qa\test.xlsx")
count = 0
for i in range(len(test.id)):
    sims = preAns(test.question[i])
    print("问题：%s"  %(test.question[i]))
    print("匹配的回答：%s" %(data.QUESTION[sims[0][0]]))
	print(sims[0:3])
    print("")
    if sims[0][0] + 1 == test.id[i]:
        count = count + 1
        
print("准确率：%10.4f" %(count/len(test.id)))			
