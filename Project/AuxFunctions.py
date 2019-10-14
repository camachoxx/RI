import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import simpleparser as parser
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import collectionloaders
import RetrievalModelsMatrix as RMM
import seaborn as sb
from itertools import cycle
import sys
from mpl_toolkits.mplot3d import Axes3D

class AuxFunctions:
    
    def __init__(self, cranfield, corpus):
        
        self.cranfield = cranfield
        self.corpus = corpus
        
    
    """
    Retorna o número médio de palavras por documento
    """
    def avg_unigrams_corpus(self):
        
        vectorizer = CountVectorizer()
        tf_cranfield = vectorizer.fit_transform(self.corpus).toarray()
        print(round(np.sum(tf_cranfield) / len(tf_cranfield)))
        return round(np.sum(tf_cranfield) / len(tf_cranfield))


    """
    #### Função para testar um modelo

    Input:
       1. modelo: nome do modelo a usar em string: "score_vsm" ; "score_lmd" ; "score_lmjm" ;
                                                   "score_rm3" ; "score_rm3+lmd" ; "score_rm3+lmjm"
       2. parâmetro do modelo: se for o modelo VSM não indicar nada
       3. bi_grams: _True_ se pretende usar bi-grams; _False_ se não (Uni-grams então)
    """
    def test_model(self, model, parameter='nan', parameter2='nan', parameter3='nan', bi_grams=False):

        if bi_grams:
            vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', 
                                            min_df=1, stop_words = {'the', 'is'})
        else:
            vectorizer = CountVectorizer()

        ### Create the model
        # Compute the term frequencies matrix and the model statistics
        tf_cranfield = vectorizer.fit_transform(self.corpus).toarray()
        models = RMM.RetrievalModelsMatrix(tf_cranfield, vectorizer)

        ### Run the queries over the corpus
        i = 1
        map_ = 0
        precL = []
        p10L=[]
        for query in self.cranfield.queries:
            # Parse the query and compute the document scores
            if model == 'score_vsm':
                scores = np.nan_to_num( models.score_vsm(parser.stemSentence(query)) )
            elif model == 'score_lmd':
                scores = np.nan_to_num( models.score_lmd(parser.stemSentence(query), parameter) )
            elif model == 'score_lmjm':
                scores = np.nan_to_num( models.score_lmjm(parser.stemSentence(query), parameter) )
            elif model == 'score_rm3+lmd':
                scores = np.nan_to_num( models.score_rm3(parser.stemSentence(query), parameter, parameter2, "lmd", parameter3) )
            elif model == 'score_rm3+lmjm':
                scores = np.nan_to_num( models.score_rm3(parser.stemSentence(query), parameter, parameter2, "lmjm", parameter3) )
            else:
                sys.exit("Nome do modelo desconhecido")

            # Do the evaluation
            [average_precision, precision, recallL, p10] = self.cranfield.eval(scores, i)
            map_ = map_ + average_precision
            precL.append(precision)
            p10L.append(p10)
            i = i + 1

        del models #deletes class instance
        map_ = map_/self.cranfield.num_queries

        return map_, recallL, precL, p10L


    """
    #### Função para treinar um modelo

    Dado um conjunto de N parâmetros a testar, corre o modelo N vezes, uma para cada parâmetro.

    Input:
       1. modelo: nome do modelo a usar em string: "score_vsm" ; "score_lmd" ; "score_lmjm" ;
                                                   "score_rm3" ; "score_rm3+lmd" ; "score_rm3+lmjm"
       2. parâmetro do modelo: se for o modelo VSM não indicar nada
       3. bi_grams: True se pretende usar bi-grams; False se não (Uni-grams então)

    Return: 6-tuplo $\rightarrow$ (map list, p10 list, rec list, prec list, best parameter, index of the best parameter)
    """
    def train_model(self, model, parameters, parameter2='nan', parameter3='nan'):

        mapL=[]; recL=[]; precL=[]; p10L=[]
        best_p = parameters[0]; best_map = 0

        for p in parameters:

            new_map, new_recall, new_precision, p10 = self.test_model(model, p, parameter2, parameter3)
            mapL.append(new_map)
            recL.append(new_recall)
            precL.append(new_precision)
            p10L.append(p10)

            if new_map > best_map:
                best_map = new_map
                best_index = np.where(parameters==p)[0][0]

        return mapL, p10L, recL, precL, best_index


    """
    #### Função fazer os gráficos de treino dos modelos para os vários pârametros

    Input:
       1. modelo: nome do modelo a usar em string: "score_vsm" ; "score_lmd" ; "score_lmjm" ;
                                                   "score_rm3" ; "score_rm3+lmd" ; "score_rm3+lmjm"
       2. titlo do gráfico
       3. lista recall
       4. lista precision
       5. lista parâmetros
       6. lista parâmetros 2 se aplicável
    """
    def plot_test(self, model, title, recall, precision, parameters, parameters2="nan"):

        #cores
        color_cycle = cycle(['dimgray','red','darkorange','goldenrod','yellow','lime',
                             'dodgerblue','darkviolet','fuchsia'])

        fig_size=[0,0]
        fig_size[0] = 7
        fig_size[1] = 7
        plt.rcParams["figure.figsize"] = fig_size
        plt.xlabel('Recall', fontsize=12, labelpad=20)
        plt.ylabel('Precision', fontsize=12, labelpad=20)
        plt.title(title, fontsize=17, pad=10)

        if parameters2 == "nan":

            for i in range(0,len(parameters)):

                if model == "score_vsm":
                    plt.plot( recall[i], np.mean(precision[i],axis=0), color = next(color_cycle), 
                         label = parameters[i] )

                else:
                    plt.plot( recall[i], np.mean(precision[i],axis=0), color=next(color_cycle), 
                         label="{0:.4f}".format(parameters[i]) )

        plt.legend(loc=1, prop={'size':17})
        plt.rc('axes', labelsize=10)
        plt.show()

        
    """
    #### Função fazer os gráficos de P@10

    Input:
       1. modelo: nome do modelo a usar em string: "score_vsm" ; "score_lmd" ; "score_lmjm" ;
                                                   "score_rm3" ; "score_rm3+lmd" ; "score_rm3+lmjm"
       2. titlo do gráfico
       3. titlo do eixo X
       4. lista P10
       5. lista parâmetros
       6. lista parâmetros 2 se aplicável
    """
    def plot_p10(self, model, title, xlabel, p10L, parameters, parameters2="nan"):

        #cores
        color_cycle = cycle(['dimgray','red','darkorange','goldenrod','yellow','lime',
                             'dodgerblue','darkviolet','fuchsia'])

        fig_size=[0,0]
        fig_size[0] = 7
        fig_size[1] = 7
        plt.rcParams["figure.figsize"] = fig_size
        plt.xlabel(xlabel, fontsize=12, labelpad=20)
        plt.ylabel('P@10', fontsize=12, labelpad=20)
        plt.title(title, fontsize=17, pad=10)

        if parameters2 == "nan":

                plt.plot(parameters, np.mean(p10L,axis=1), 'o', markersize=8, color=next(color_cycle))
                plt.plot(parameters, np.mean(p10L,axis=1), ':')

        #else:
        #   for i in range(0,len(parameters)):
        #      for j in range(0,len(parameters)):
        #plt.gca().set_ylim(0,1)
        plt.show()
        
        
    """
    #### Função fazer a curva Precision-Recall do melhor modelo do treino

    Input:
       1. modelo: nome do modelo a usar em string: "score_vsm" ; "score_lmd" ; "score_lmjm" ;
                                                   "score_rm3" ; "score_rm3+lmd" ; "score_rm3+lmjm"
       2. titlo do gráfico
       3. map
       4. lista precision
       5. lista recall
    """        
    def best_test(self, model, title, map_, prec, rec):

        fig_size=[0,0]
        fig_size[0] = 7
        fig_size[1] = 7
        plt.rcParams["figure.figsize"] = fig_size

        print('MAP=',map_)

        plt.plot(rec, np.mean(prec,axis=0), color='b', alpha=1)
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.fill_between(rec, 
                     np.mean(prec,axis=0)-np.std(prec,axis=0), 
                     np.mean(prec,axis=0)+np.std(prec,axis=0), facecolor='b', alpha=0.1)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.gca().set_xlim(0,1)
        plt.title(title)
        plt.show()
        
        
        """
        #### Função fazer as curva Precision-Recall de todos os modelos

        Input:
           1. titlo do gráfico
           2. número de modelos
           3. lista legenda
           4. lista precision
           5. lista recall
        """  
        def plot_PR_all_models(self, title, nr_models, legend, recall, precision):

            #cores
            color_cycle = cycle(['dimgray','red','yellow','lime',
                                 'dodgerblue','darkviolet','fuchsia'])

            fig_size=[0,0]
            fig_size[0] = 7
            fig_size[1] = 7
            plt.rcParams["figure.figsize"] = fig_size
            plt.xlabel('Recall', fontsize=12, labelpad=20)
            plt.ylabel('Precision', fontsize=12, labelpad=20)
            plt.title(title, fontsize=17, pad=10)

            for i in range(nr_models):

                plt.plot( recall[i], np.mean(precision[i],axis=0), color=next(color_cycle), 
                             label=legend[i] )

            plt.legend(loc=1, prop={'size':15})
            plt.rc('axes', labelsize=10)
            plt.show()