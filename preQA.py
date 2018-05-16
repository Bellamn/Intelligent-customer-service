from textrank4zh import TextRank4Keyword, TextRank4Sentence
import pandas as pd
from gensim import corpora, models, similarities

tr4w = TextRank4Keyword()
data = pd.read_excel(r"C:\Users\dyh\Desktop\qa\FAQ.xlsx")
    
text = "?".join(data.QUESTION)
tr4w.analyze(text=text, lower=True, window=2)
sentence_q = tr4w.words_no_stop_words
dictionary_q = corpora.Dictionary(sentence_q)
#dictionary.save(r"C:\Users\dyh\Desktop\qa\deerwester.dict")
corpus_q = [dictionary_q.doc2bow(text) for text in sentence_q]
#corpora.MmCorpus.serialize('/home/dyh/Desktop/zhineng/deerwester.mm', corpus)

tfidf_q = models.TfidfModel(corpus_q)
corpus_tfidf_q = tfidf_q[corpus_q]
lsi_q = models.LsiModel(corpus_tfidf_q, id2word=dictionary_q, num_topics=20)
#lsi.save('/home/dyh/Desktop/zhineng/model.lsi')
index_q = similarities.MatrixSimilarity(lsi_q[corpus_q])
def preBaseQ(new_doc):
    tr4w.analyze(text= new_doc, lower=True, window=2)
    new_doc =  tr4w.words_no_stop_words 
    new_vec = dictionary_q.doc2bow(new_doc[0])
    new_lsi = lsi_q[new_vec]
    sims = index_q[new_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims



sentence_qa = []
for i in range(len(data.ID)):
    temp = []
    str = data.QUESTION[i]+data.ANSWER[i]
    tr4w.analyze(text=str, lower=True, window=2)
    word = tr4w.words_no_stop_words
    for each in word:
        temp = temp + each
    sentence_qa.append(temp)
dictionary_qa = corpora.Dictionary(sentence_qa)
#dictionary.save(r"C:\Users\dyh\Desktop\qa\deerwester.dict")
corpus_qa = [dictionary.doc2bow(text) for text in sentence_qa]
#corpora.MmCorpus.serialize('/home/dyh/Desktop/zhineng/deerwester.mm', corpus)

tfidf_qa = models.TfidfModel(corpus_qa)
corpus_tfidf_qa = tfidf_qa[corpus_qa]
lsi_qa = models.LsiModel(corpus_tfidf_qa, id2word=dictionary_qa, num_topics=20)
#lsi.save('/home/dyh/Desktop/zhineng/model.lsi')

index_qa = similarities.MatrixSimilarity(lsi_qa[corpus_qa])
    
def preBaseQA(new_doc):  
    tr4w.analyze(text= new_doc, lower=True, window=2)
    new_doc =  tr4w.words_no_stop_words 
    new_vec = dictionary_qa.doc2bow(new_doc[0])
    new_lsi = lsi_qa[new_vec]
    sims = index_qa[new_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims
    
active = True
while active:
    str = input()
    if str == 'quit':
        active = False
    else : 
        sims = preAns(str)
        if sims[0][1] < 0.7:
            print("请详细描述您的问题")
        else : 
            print(data.QUESTION[sims[0][0]])
            print(data.ANSWER[sims[0][0]])
            print(sims[0:3])
            
test = pd.read_excel(r"C:\Users\dyh\Desktop\qa\test.xlsx")
count = 0
for i in range(len(test.id)):
    sims_q = preBaseQ(test.question[i])
    sims_qa = preBaseQA(test.question[i])
    print("问题：%s 期望答案：%d"  %(test.question[i], test.id[i]-1))
    print("匹配的回答：%s" %(data.QUESTION[sims_qa[0][0]]))
    print(sims_qa[0:3])
    print(sims_q[0:3])
    print("")
    if sims_q[0][0] + 1 == test.id[i]:
        count = count + 1        
print("准确率：%10.4f" %(count/len(test.id)))