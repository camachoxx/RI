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
        
        #frequencia de cada token no corpus ---> Mc(t)
        self.Mc = np.sum(self.tf,axis=0) / self.total_tokens_corpus
        
        
        ############ LMJM statistics ############
        self.doc_col=vectorizer.get_feature_names()
        self.doc_df=pd.DataFrame(tf,columns=self.doc_col)
        
        
        ############ BM25 statistics ############

        
    def score_vsm(self, query):
        
        query_vector = self.vectorizer.transform([query]).toarray()
        query_norm = np.sqrt(np.sum(np.power(query_vector, 2), axis=1))

        doc_scores = np.dot(query_vector, self.tfidf.T) / (0.0001 + self.docNorms * query_norm)
#        doc_scores = np.nan_to_num(doc_scores, 0)

        return doc_scores

    
    def score_lmd(self, query, miu):
        
        query_vector = self.vectorizer.transform([query]).toarray()
        
        #probabilidades ---> P(t|Md,Mc)
        p = (self.tf + miu*self.Mc)/(self.docLen[:,None] + miu)
        
        #elevar ao vetor da query ---> P(t|Md,Mc) ^ q(t)
        e = np.power(p, query_vector)
        
        #produto das probabilidades ---> produtorio( P(t|Md,Mc)^ q(t) )
        doc_scores = np.prod(e, axis=1)
        
        return doc_scores
    

    def score_lmjm(self, query, lam):
        
        query_vector = self.vectorizer.transform([query]).toarray()
        
        #probabilidades
        Pd = self.tf / self.docLen[:,None]
        Pc = self.Mc

        #elevar ao vetor da query (query_vector)
        Ed = np.power(Pd, query_vector)
        Ec = np.power(Pc, query_vector)
        
        #produto das probabilidades
        doc_scores = np.prod(lam*Ed + (1-lam)*Ec, axis=1)
        
        return doc_scores
        
    """    
    def lmjm_aux(self,y,pd,pc):
        v=pd*y+pc*(1-y)
        return v
    
    def score_lmjm(self, query,y,d):
        #Only works for d document for now
        query_df=pd.DataFrame(self.vectorizer.transform([query]).toarray() , columns=self.doc_col)
        cl=query_df[query_df!=0].dropna(axis=1).columns
        mul=1
        for i in query_df[query_df!=0].dropna(axis=1).columns:
                print(i)
                Pd=np.sum(self.doc_df.loc[d,i])/np.sum(self.doc_df.iloc[d,:])
                print(Pd)
                Pc=np.sum(self.doc_df.loc[:,i])/self.total_tokens_corpus
                print(Pc)
                mul=mul*self.lmjm_aux(y,Pd,Pc)
                
        return mul
    """                                                           
    
                                                                   
    def score_bm25(self, query):
        return 0

    def scoreRM3(self, query):
        return 0

