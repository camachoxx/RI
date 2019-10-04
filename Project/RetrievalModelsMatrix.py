import numpy as np


class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer):
        self.vectorizer = vectorizer
        self.tf = tf

        ## VSM statistics
        self.term_doc_freq = np.sum(tf != 0, axis=0)
        self.term_coll_freq = np.sum(tf, axis=0)
        self.docLen = np.sum(tf, axis=1)

        self.idf = np.log(np.size(tf, axis = 0) / self.term_doc_freq)
        self.tfidf = np.array(tf * self.idf)

        self.docNorms = np.sqrt(np.sum(np.power(self.tfidf, 2), axis=1))

        ## LMD statistics
        self.colection_Total=np.sum(self.docLen)
        
        ## LMJM statistics
        
        
        ## BM25 statistics

        
    def score_vsm(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (0.0001 + self.docNorms * query_norm)
#        doc_scores = np.nan_to_num(doc_scores, 0)

        return doc_scores

    def score_lmd(self, query,u):
        query_vector = self.vectorizer.transform([query]).toarray()
        
        doc_freq=np.dot(query_vector,self.tf.T)
        
        doc=(doc_freq + u * np.sum(doc_freq)/self.colection_Total) / (self.colection_Total + u)
        
        doc_scores=np.prod(doc**doc_freq)
        return doc_scores

    def score_lmjm(self, query):
        return 0

    def score_bm25(self, query):
        return 0

    def scoreRM3(self, query):
        return 0

