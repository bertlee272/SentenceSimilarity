# SentenceSimilarity
Comparing Sentence Embedding Models

## 句子相似度計算模型比較 Sentence Similarity Model Comparison

Data Source: [螞蟻金融NLP競賽](https://dc.cloud.alipay.com/index?click_from=MAIL&_bdType=acafbbbiahdahhadhiih#/topic/intro?id=3)

- 使用三種Model生成Sentence Vector:
	- TF-IDF ([sklearn-TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html))
	- Word2Vec ([使用新聞、百度百科、小說數據來訓練的64維的Word2Vec模型](https://www.floydhub.com/cliffk321/datasets/news_12g_baidubaike_20g_novel_90g_embedding_64bin))
	- BERT ([bert-as-service](https://github.com/hanxiao/bert-as-service))
- 計算Sentence Pair的相似度
- 設定Threshold值，輸出同義/不同義
- 比較判斷準確度Accuracy

![alt text](SentenceSimilarityCalculate.png "TFIDF")
![alt text](句子相似度比較.png "SentSimilar")