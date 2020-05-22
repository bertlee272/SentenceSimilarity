import numpy as np
import pandas as pd
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Calculate tfidf similarity
def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    # 將字中間加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 轉化為TF矩陣
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    cc = TfidfVectorizer(tokenizer=lambda s: s.split()[0])
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    vector = cc.fit_transform(corpus).toarray()
    # 計算TF系數
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

# Calculate label predicting accuracy
def calAccuracy(df, threshold):
    label = df['label']
    sim = df['tfidfSimilarity']
    dataSize = len(label)
    accurateCount = 0
    for i in range(dataSize):
        if (label[i]==1 and sim[i]>threshold/100):
            accurateCount += 1
        elif (label[i]==0 and sim[i]<threshold/100):
            accurateCount += 1
    accuracy = accurateCount/dataSize
    return accuracy

# -------------------------
if __name__ == "__main__":
	# Read file
	dfTest = pd.read_csv('realHwaBei.csv', encoding='gbk',  header=None, names=['idx','sentence1','sentence2','label'])
	sents1 = dfTest['sentence1']
	sents2 = dfTest['sentence2']
	dataSize = len(sents1)
	tfidfSim = ["0" for x in range(dataSize)]
	startTime = time.time()

	for i in range(dataSize):
	    tfidfSim[i] = tfidf_similarity(sents1[i],sents2[i])
	    timeNow = time.time()
	    pastTime = timeNow-startTime
	    print('{} | {:0.1f} | {:0.3f}'.format(i, pastTime, tfidfSim[i]))

	dfTest['tfidfSimilarity'] = tfidfSim
	# Save tfidf similarity
	dfTest.to_csv('resultHB.csv', encoding="utf_8_sig", index=False)

	# Read file w/ tfidf similarity
	dfResult = pd.read_csv('resultHB.csv', encoding='utf-8')

	# Calculate prediction accuracy for each labeling threshold
	for threshold in range(1,100):
	    acc = calAccuracy(dfResult,threshold)
	    if acc>0.1:
	        print('{:0.3} | {}'.format(acc,threshold/100))
