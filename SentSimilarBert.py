from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
import pandas as pd


# Calculate label predicting accuracy
def calAccuracy(df, threshold):
	label = df['label']
	sim = df['bertSimilarity']
	dataSize = len(label)
	accurateCount = 0
	for i in range(dataSize):
		if (label[i]==1 and sim[i]>threshold/1000):
			accurateCount += 1
		elif (label[i]==0 and sim[i]<threshold/1000):
			accurateCount += 1
	accuracy = accurateCount/dataSize
	return accuracy

if __name__ == "__main__":
	# Have to connect to BERT Server first
	# Using Bert server
	bc = BertClient(check_length=False)

	# Read file
	dfTest = pd.read_csv('resultHB.csv')
	sents1 = dfTest['sentence1']
	sents2 = dfTest['sentence2']
	dataSize = len(sents1)
	bertSim = ["0" for x in range(dataSize)]
	startTime = time.time()

	for i in range(dataSize):
		# Generating BERT sentence Vector
		sent1 = bc.encode([sents1[i]])
		sent2 = bc.encode([sents2[i]])
		# Calculate Similarity
		sim = cosine_similarity(sent1.reshape(-1, 768), sent2)
		bertSim[i] = sim[0][0]
		timeNow = time.time()
		pastTime = timeNow-startTime
		print('{} | {:0.1f} | {:0.3f}'.format(i, pastTime, bertSim[i]))

	dfTest['bertSimilarity'] = bertSim
	# Save file w/ BERT Vector Similarity
	dfTest.to_csv('resultHB.csv', encoding="utf_8_sig", index=False)




	# Read file w/ BERT similarity
	dfResult = pd.read_csv('resultHB.csv', encoding='utf-8')
	# Calculate prediction accuracy for each labeling threshold
	for threshold in range(750,1000):
		acc = calAccuracy(dfResult,threshold)
		if acc>0.1:
			print('{:0.3} | {}'.format(acc,threshold/1000))
