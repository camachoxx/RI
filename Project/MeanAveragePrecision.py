import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



class MeanAveragePrecision:
    def __init__(self, tf, vectorizer):
        self.vectorizer = vectorizer
        self.tf = tf
        self.termCollFreq = np.sum(tf, axis=0)
        self.Len = tf.shape[0]
        
    def scoreMap(self,query):
        
        query_vector = self.vectorizer.transform([query]).toarray()
        
        doc_scores=np.dot(query_vector,self.tf.T)/self.Len
        return doc_scores
        
    def cumulative_pscore(self,query):
        query_vector = self.vectorizer.transform([query]).toarray()
        vect=np.arange(1,self.Len + 1)
        doc_scores=np.cumsum(np.dot(query_vector,self.tf.T))/vect
        return doc_scores
    
    def recall(self,query):
        query_vector = self.vectorizer.transform([query]).toarray()
        docf=np.dot(query_vector,self.tf.T)
        vect=np.sum(docf)
        doc_scores=np.cumsum(docf)/vect
        return doc_scores
    
    def pr_curve(self,query):
        p=self.cumulative_pscore(query)
        r=self.recall(query)
        
        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend("Precision-Recall curve",title=query)
        
        return plt.plot(r,p);
        