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

    
    def score_lmd(self, query, mu, query_to_vector=True):
        
        if query_to_vector:
            query_vector = self.vectorizer.transform([query]).toarray()
        else:
            query_vector = query
        
        #probabilidades ---> P(t|Md,Mc)
        p = (self.tf + mu*self.Mc)/(self.docLen[:,None] + mu)
        
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


    def score_rm3(self, query, alfa, top_docs, model, parameter, query_to_vector=True):

        top_words = 20 #escolher o nr de palavras a considerar ---> top_words
        
        if query_to_vector:
            query_vector = self.vectorizer.transform([query]).toarray()
        else:
            query_vector = query
        
        #scores iniciais
        if model == 'lmd':
            initial_scores = self.score_lmd(query_vector, parameter, query_to_vector=False)
        else: #model == 'lmjm'
            initial_scores = self.score_lmjm(query_vector, parameter, query_to_vector=False)
        
        #ordena os doc_scores do menor para o maior
        sorted_scores = np.sort(initial_scores)
        
        #minimo das probabilidades dos documentos relevante
        threshold = sorted_scores[-1*top_docs]
        
        #documentos mais relevantes
        best_scores = initial_scores * (initial_scores >= threshold)
        
        #probabilidade no modelo RM1 contabilizando todos as palavras
        P_RM1_all_words = np.sum(self.Md * best_scores[:,None], axis=0)

        sorted_words = np.sort(P_RM1_all_words)

        words_threshold = sorted_words[-1*top_words]

        P_RM1 = P_RM1_all_words * (P_RM1_all_words >= words_threshold)

        new_query = (1-alfa)*query_vector + alfa*P_RM1
        
        #scores finais
        if model == 'lmd':
            doc_scores = self.score_lmd(new_query, parameter, query_to_vector=False)
        else: #model == 'lmjm'
            doc_scores = self.score_lmjm(new_query, parameter, query_to_vector=False)
        
        return doc_scores

