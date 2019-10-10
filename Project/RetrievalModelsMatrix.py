import numpy as np
import pandas as pd

class RetrievalModelsMatrix:

    def __init__(self, tf, vectorizer):
        
        self.vectorizer = vectorizer
        self.tf = tf

        
        ############ VSM statistics ############
        self.term_doc_freq = np.sum(tf != 0, axis=0)
        self.term_coll_freq = np.sum(tf, axis=0)
        
        #numero de tokens em cada documento ---> |d|
        self.docLen = np.sum(tf, axis=1)

        self.idf = np.log(np.size(tf, axis = 0) / self.term_doc_freq)
        self.tfidf = np.array(tf * self.idf)

        self.docNorms = np.sqrt(np.sum(np.power(self.tfidf, 2), axis=1))

        
        ############ LMD statistics ############
        
        #numero de tokens no corpus
        self.total_tokens_corpus = np.sum(self.docLen)
        
        #probabilidade de cada token no corpus ---> P(t|Mc)
        self.Mc = np.sum(self.tf,axis=0) / self.total_tokens_corpus
        
        
        ############ LMJM statistics ############
        #probabilidade de uma palavra num doc ---> P(t|Md)
        self.Md = self.tf / (self.docLen[:,None] + 0.0001)
        
        
        ############ BM25 statistics ############

        
    def score_vsm(self, query, query_to_vector=True):
        
        if query_to_vector:
            query_vector = self.vectorizer.transform([query]).toarray()
        else:
            query_vector = query
            
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (0.0001 + self.docNorms * query_norm)
#        doc_scores = np.nan_to_num(doc_scores, 0)

        return doc_scores

    
    def score_lmd(self, query, miu, query_to_vector=True):
        
        if query_to_vector:
            query_vector = self.vectorizer.transform([query]).toarray()
        else:
            query_vector = query
        
        #probabilidades ---> P(t|Md,Mc)
        p = (self.tf + miu*self.Mc)/(self.docLen[:,None] + miu)
        
        #elevar ao vetor da query ---> P(t|Md,Mc) ^ q(t)
        e = np.power(p, query_vector)
        
        #produto das probabilidades ---> produtorio( P(t|Md,Mc)^ q(t) )
        doc_scores = np.prod(e, axis=1)
        
        return doc_scores
    

    def score_lmjm(self, query, lam, query_to_vector=True):
        
        if query_to_vector:
            query_vector = self.vectorizer.transform([query]).toarray()
        else:
            query_vector = query

        #elevar ao vetor da query (query_vector)
        Ed = np.power(self.Md, query_vector)
        Ec = np.power(self.Mc, query_vector)
        
        #produto das probabilidades
        doc_scores = np.longdouble(np.prod(lam*Ed + (1-lam)*Ec, axis=1))
        
        return doc_scores
        
        
    def score_bm25(self, query):
        """a=self.tf
        b=self.docLen[:,None]
        #probabilidades
        with np.errstate(divide='ignore', invalid='ignore'):
            Pd = np.true_divide( a, b )
            Pd[ ~ np.isfinite( Pd )] = 0  # -inf inf NaN
        #Pd = self.tf / self.docLen[:,None]"""
        return 0

    
    def score_rm3(self, query, alfa, top_docs, model, parameter, query_to_vector=True):
        """i: indica qual o modelo inicial:  i=1 --> lmjm"""
        
        if query_to_vector:
            query_vector = self.vectorizer.transform([query]).toarray()
        else:
            query_vector = query
            
        #doc_scores
        initial_scores = model(query_vector, parameter, query_to_vector=False)
        
        #ordena os doc_scores do menor para o maior
        sorted_scores = np.sort(initial_scores)
        
        #minimo das probabilidades dos documentos relevante
        threshold = sorted_scores[-1*top_docs]
        
        #documentos mais relevantes
        best_scores = initial_scores * (initial_scores >= threshold)
        
        #probabilidade no modelo RM1
        P_RM1 = np.sum(self.Md * best_scores[:,None], axis=0)
        
        new_query = (1-alfa)*query_vector + alfa*P_RM1
        
        doc_scores = model(new_query, parameter, query_to_vector=False)
        
        return doc_scores

