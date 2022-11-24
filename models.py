#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from utils import sentence_preprocessing
from utils import fusion
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

 

class Basic_model():
    """
        cette class permet de représenter des models dit basiques qui font des prédictions grâce à l'algorithme knn pour 
        un certain embedding choisi. Un tel model est défini par
            - vectorizer : represente l'objet permettant de réalisé l'embedding
            - k est le nombre de documents les plus susceptibles de répondre à la question
            - context_id : la liste d'identifiant des contexts à prédire
            - context_embedding : qui est l'embedding du context sur lequel on doit réaliser la prédiction
            - predictor qui est le knn qui réalise la prédiction
    """
    def __init__(self, context_embedding, context_id, k, vectorizer, n_jobs = -1):
        self.vectorizer = vectorizer
        self.k = k
        self.context_id = np.array(context_id)
        self.context_embedding = context_embedding
        self.predictor = NearestNeighbors(n_neighbors = self.k, metric ='cosine', n_jobs = n_jobs)
  
    
    def fit(self):
        self.predictor.fit(self.context_embedding, self.context_id)
        
    def predict(self, question_embedding, return_score = False):
        """
            cette methode renvoit la liste des index de k meilleurs documents qui réponde à la question
            elle retourne aussi les distance suivant la métrique cosine entre la question et les différents documents
        """
          
        if return_score:
            pred_dist, pred_pos = self.predictor.kneighbors(question_embedding, return_distance = return_score)
            pred_context = self.context_id[pred_pos.flatten()].reshape(len(pred_pos), -1)
            return pred_dist, pred_context
        
        else:
            pred_pos = self.predictor.kneighbors(question_embedding, return_distance = return_score)
            pred_context = self.context_id[pred_pos.flatten()].reshape(len(pred_pos), -1)
            return pred_context
    

class Tfidf_model(Basic_model):
    def __init__(self, context_embedding, context_id, k, vectorizer = 'default', embedding_done  = False, n_jobs = -1):
        if vectorizer == 'default':
            # la méthode d'embedding choisit par défaut
            vectorizer = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english', max_features=80000, binary = True)
        
        if not embedding_done:
            # réalise l'embedding des contexts si ce n'est pas encore fait
            context_embedding = vectorizer.fit_transform(context_embedding)
            
        super().__init__(context_embedding, context_id, k, vectorizer, n_jobs = -1)
    
    def predict(self, question, embedding_done = False, return_score = False):
        # on réalise l'embedding de la question passée en paramètre si ce n'est pas encore le cas
        if embedding_done:
            question_embedding = question
        else:
            question_embedding = self.vectorizer.transform(question)
        
        return super().predict(question_embedding, return_score)
    

class Transformer_model(Basic_model):
    def __init__(self, context_embedding, context_id, k, model = 'default', embedding_done  = True, n_jobs = -1):
        # Par défaut l'embedding des contexts est choisit comme étant déja fait car c'est couteux en temps de le réaliser
        if model == 'default':
            model = SentenceTransformer('all-MiniLM-L6-v2')
        if not embedding_done:
            context_embedding = np.apply_along_axis(lambda x: model.encode(x),0,context_embedding)
            
        super().__init__(context_embedding, context_id, k, model, n_jobs = -1)
    
    def predict(self, question, embedding_done = False, return_score = False):
        if embedding_done:
            question_embedding = question
        else:
            question_embedding = np.apply_along_axis(lambda x: self.vectorizer.encode(x),0,question)
        
        return super().predict(question_embedding, return_score)


class BM25_model():
    """
    model de prédiction basé sur bm25
    """
    def __init__(self, context_preprocessed, context_id, k, preprocessing_done = False):
        if not preprocessing_done:
            context_preprocessed = pd.DataFrame(context_preprocessed).iloc[:,0]
            self.context_preprocessed = context_preprocessed.apply(lambda x: sentence_preprocessing(x))
        else:
            self.context_preprocessed = context_preprocessed
        self.k = k
        self.bm25 = BM25Okapi(self.context_preprocessed)
        self.context_id = np.array(context_id)
    
    def fit(self):
        return
    
    def predict(self, question, batch_size = 1000, preprocessing_done = False, return_score = False, verbose = False):
        if not preprocessing_done:
            question = pd.DataFrame(question).iloc[:,0]
            tokenized_question = question.apply(lambda x: sentence_preprocessing(x))
        else:
            tokenized_question = question
        
        score_context = np.stack(tokenized_question[0: batch_size].apply(lambda x: -self.bm25.get_scores(x)), axis = 0)
        n = len(tokenized_question)
        # on sépare l'ensemble de question en batch de taille batch_size et on fait progressivement la prédiction
        # sur chaque batch
        # on calcul l'oppossé du score pour pouvoir réaliser au final une 
        if n % batch_size == 0:
            total_batch = n//batch_size
        else:
            total_batch = n//batch_size + 1
        if verbose:
            print("batch 1 done over " + str(total_batch))
            
        for i in range(1,total_batch):    
            tmp = tokenized_question[i*batch_size: min(batch_size*(i+1),n)].apply(lambda x: -self.bm25.get_scores(x))
            score_context = np.concatenate((score_context,np.stack(tmp)), axis = 0)
            if verbose:
                print("batch " + str(i) + " done over " + str(total_batch))
        
        # on récupere les indices des k meilleurs scores 
        top_context = np.argpartition(score_context, kth=self.k, axis = 1)
        top_context = top_context[:,:self.k]
        #on récupere les k meilleurs scores et on les tri
        score_context = np.take_along_axis(score_context, top_context, axis = 1)
        tmp_position = np.argsort(score_context, axis = 1)
        #on récupere les k meilleurs contexts
        best_position = np.take_along_axis(top_context, tmp_position, axis = 1)
        pred_context = self.context_id[best_position.flatten()].reshape(len(best_position), -1)
        
        if return_score:
            score_context = - np.take_along_axis(score_context, tmp_position, axis = 1)
            return score_context, pred_context
        else:
            return pred_context
        
        
class Combined_model():
    """
        cette class permet de créer un model obtenu en combinant deux models. elle est définit par 
            - ses deux models crées chacun à l'aide d'un dictionnaire de ses paramètres
            - cross_encoder qui est le cross encoder utiliser pour selectionner les k meilleurs documents
            - k le nombre de meilleurs documents
    """
    def __init__(self, context_id, context_text, model1_params, model2_params, k, cross_encoder = 'default',n_jobs = -1):
        
        
        if cross_encoder == 'default':
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-4-v2')
            
        self.cross_encoder = cross_encoder
        self.context_id = np.array(context_id)
        self.context_text = context_text
        self.k = k
        self.model1 = self.get_model(model1_params, context_text)
        self.model2 = self.get_model(model2_params, context_text)
        
    def predict_query(self,query):
        return np.array([self.cross_encoder.predict([query[0], elt]) for elt in query[1:]])
    
    def nb_context_to_get(self):
        if self.k <= 10:
            return 20
        else:
            return self.k
    
    def get_model(self, model_param, context_text):
        """
            permet de créer un model à l'aide de ses paramètres et de l'ensemble des contextes sur lequel on réalise
            la prédiction
        """
        
        nb = self.nb_context_to_get()
        if model_param['name'] == 'tfidf':
            vectorizer = model_param['vectorizer']
            model = Tfidf_model(context_text, self.context_id, nb, vectorizer)
        
        elif model_param['name'] == 'transformer':
            vectorizer = model_param['vectorizer']
            context_embedding = model_param['context_embedding']
            model = Transformer_model(context_embedding, self.context_id, nb, vectorizer)
        
        elif model_param['name'] == 'bm25':
            model = BM25_model(context_text, self.context_id, nb)
        
        else:
            print('error model name')
            return
        
        return model
    
    
    def fit(self):
        self.model1.fit()
        self.model2.fit()
        return
    
    def predict(self, questions, method = 'rep', batch_size = 1000, verbose = False):
        
        # prédiction suivant la méthode choisie
        if method == 'score':
            y_score_1, y_pred_1 = self.model1.predict(questions, return_score = True)
            y_score_2, y_pred_2 = self.model2.predict(questions, return_score = True)
            y_final = fusion(y_pred_1,y_pred_2, method = method, y_score_1 = y_score_1, y_score_2 = y_score_2)
            
        elif method == 'basic' or method == 'rep':
            y_pred_1 = self.model1.predict(questions)
            y_pred_2 = self.model2.predict(questions)            
            y_final = fusion(y_pred_1,y_pred_2, method= method)    
        
        # on fusionne chaque question et l'ensemble de ses contextes possibles en ayant pour chaque question
        # une liste qui contient en position 0 le text de la question et le reste de la liste contient les texts des meilleurs
        # documents obtenus grace aux deux models qui le compose
        top_context = np.array(self.context_text.loc[y_final.flatten()]).reshape(len(y_final), -1)
        questions = np.array(questions)
        queries = np.concatenate((questions.reshape(-1,1), top_context), axis = 1)
        
        scores = np.apply_along_axis(lambda x: self.predict_query(x), 1, queries[:batch_size])
        n = len(queries)
        
        # on fait des prédictions par batch de questions
        if n % batch_size == 0:
            total_batch = n//batch_size
        else:
            total_batch = n//batch_size + 1
        if verbose:
            print("batch 1 done over " + str(total_batch))
            
        for i in range(1,total_batch):    
            tmp = np.apply_along_axis(lambda x: self.predict_query(x), 1, queries[i*batch_size: min(batch_size*(i+1),n)])
            scores = np.concatenate((scores,tmp), axis = 0)
            if verbose:
                print("batch " + str(i) + " done over " + str(total_batch))
        
        # on récupère les k meilleurs documents
        top_context = np.argpartition(-scores, kth=self.k, axis = 1)
        y_final = np.take_along_axis(y_final, top_context, axis=1)[:,:self.k]
        scores = np.take_along_axis(scores, top_context, axis=1)[:,:self.k]
        ordered_scores_position = np.argsort(-scores, axis = 1)
        final_pred = np.take_along_axis(y_final, ordered_scores_position, axis=1)
        
            
        return final_pred
    
    def get_context(self,y):
        return self.context_text.loc[y]
                   
        


# In[405]:




