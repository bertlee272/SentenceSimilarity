import gensim
import jieba
import numpy as np
from scipy.linalg import norm
import pandas as pd
import time

t1 = time.time()
model_file = 'news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
t2 = time.time()
print('load wordVector time:{:0.3}'.format(t2-t1))


# calculate Sentence Vector(avg of word vector) similarity
def vector_similarity(s1, s2):
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(64)
        for word in words:
            if word in model:
                print(word)
                v += model[word]
        	# else:
        	# 	v += model[]
        v /= len(words)
        return v
    
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


#read file
dfTest = pd.read_csv('resultHB.csv', encoding='utf-8')
sents1 = dfTest['sentence1']
sents2 = dfTest['sentence2']
dataSize = len(sents1)
w2vSim = ["0" for x in range(dataSize)]
startTime = time.time()

for i in range(dataSize):
	w2vSim[i] = vector_similarity(sents1[i],sents2[i])
	timeNow = time.time()
	pastTime = timeNow-startTime
	print('{} | {:0.1f} | {:0.3f}'.format(i, pastTime, w2vSim[i]))

dfTest['w2vSimilarity'] = w2vSim
#save sentence vector similarity
dfTest.to_csv('resultHB.csv', encoding="utf_8_sig", index=False)


# calculate label predicting accuracy
def calAccuracy(df, threshold):
	label = df['label']
	sim = df['w2vSimilarity']
	dataSize = len(label)
	accurateCount = 0
	for i in range(dataSize):
		if (label[i]==1 and sim[i]>threshold/100):
			accurateCount += 1
		elif (label[i]==0 and sim[i]<threshold/100):
			accurateCount += 1
	accuracy = accurateCount/dataSize
	return accuracy

# read file w/ sent2vec similarity
dfResult = pd.read_csv('resultHB.csv', encoding='utf-8')

# calculate prediction accuracy for each labeling threshold
for threshold in range(1,100):
	acc = calAccuracy(dfResult,threshold)
	if acc>0.1:
		print('{:0.3} | {}'.format(acc,threshold/100))