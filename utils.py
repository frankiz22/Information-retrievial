#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def get_data(file_name, name):
    """
        permet de recuperer les données dans un fichier .json
        retourne deux datasets:
            - un dataset X_context qui contient les contexts et leurs identifiants id_context
            - un data set qui contient les questions et l'identifiant du context qui lui est lié
    """
   
    f = open(file_name)    
    data_dict = json.load(f)
    f.close()
    
    X_question = pd.DataFrame(columns=['question','id_context'])
    X_context = pd.DataFrame(columns=['id_context','text_context'])
    
    
    nb_context = 0 # ce compteur nous permet d'affecter les identifiants des contexts

    for text in data_dict['data']:
        for paragraph in text['paragraphs']:
            X_context = X_context.append({'id_context':nb_context, 'text_context':paragraph['context']}, ignore_index= True)
            for qas in paragraph['qas']:
                X_question = X_question.append({'question':qas['question'], 'id_context':nb_context},ignore_index=True)
            nb_context += 1            
            
  
    X_question.to_csv(name + '_questions.csv', index = False)
    X_context.to_csv(name +'_contexts.csv', index = False)
    
    return X_question,X_context


def stemming(text, stemmer = SnowballStemmer(language='english')):
    """
        retourne une phrase où chaque mot a préprocessé à l'aide du stemming
    """
    
    my_list = []
    for token in word_tokenize(text):
        my_list.append(stemmer.stem(token))
    return ' '.join(my_list)


def lemmatization(text, lemmatizer = WordNetLemmatizer()):
    """
        retourne une phrase où chaque mot a préprocessé à l'aide de la lemmatization
    """
    my_list = []
    for token in word_tokenize(text):
        my_list.append(lemmatizer.lemmatize(token))
    return ' '.join(my_list)
    


def sentence_preprocessing(sentence, tokenizer = RegexpTokenizer(r"\w+")):
    """
        pour une phrase donnée, la fonction :
            - retire les ponstuations
            - separe la phrase en différents tokens
            - les mets tous en miniscules
            - retires les stops words
    """
    word_list = tokenizer.tokenize(sentence)
    word_list = np.char.lower(word_list)
    return word_list[~np.in1d(word_list, stopwords.words('english'))]

def accuracy(y_true, y_pred, k):
    """
        retourne le top -k accuracy
    """
    
    y_true = np.array(y_true).reshape(-1,1)
    # la valeur exacte est mise en avant de la liste des prédiction
    # pour chaque échantillon ainsi formé, on vérifie si l'élément en position 0(la vraie valeur)
    # est dans la liste des prédcitions (le reste de la liste en partant de la position 1)
    y_compare = np.concatenate((y_true,y_pred), axis = 1)
    
    return np.mean(np.apply_along_axis(lambda x: x[0] in x[1:k+1], 1, y_compare))



def fusion_basic(y_pred_1, y_pred_2):
    """
        retourne la concatenation des prédictions.
        les deux prédictions sont les top k obtenues à partir de deux méthodes différentes
        la fonction renvoit un top k obtenus à parti du top k//2 des deux méthodes
        si k est impaires on considère le top k//2 + 1 de la première prédiction
    """
    k = len(y_pred_1[0])
    
    return np.concatenate((y_pred_1[:,:k//2 + 1*(k%2)],y_pred_2[:,:k//2]), axis = 1)

def fusion_with_score(y_pred_1, y_score_1, y_pred_2, y_score_2):
    """
        prend en paramètre les prédictions top-k obtenues grace à deux méthodes ainsi que les scores associés
        elle permet de faire la fusion des résultats des deux méthodes en considérant les k premières prédictions
        avec les meilleurs scores
    """
    
    # les scores sont normalisés pour chaque échantillon
    y_score_1 = y_score_1/y_score_1.sum(axis = 1).reshape(-1,1)
    y_score_2 = y_score_2/y_score_2.sum(axis = 1).reshape(-1,1)
    y_final = []
    k = len(y_pred_1[0])
    
    for i in range(len(y_pred_1)):
        tmp = []
        dic = {}
        ib = 0
        it = 0
        # à chaque fois qu'on choisi le context avec le meilleur score actuel, on vérifie s'il a déjà ata selectionné
        # précedemment
        while(len(tmp) < k and ib < len(y_score_1[i]) and it < len(y_score_2[i])):
            if y_score_1[i][ib] > y_score_2[i][it]:
                if y_pred_2[i][it] not in dic:
                    tmp.append(y_pred_2[i][it])
                    dic[y_pred_2[i][it]] = True
                it+= 1
            else:
                if y_pred_1[i][ib] not in dic:
                    tmp.append(y_pred_1[i][ib])
                    dic[y_pred_1[i][ib]] = True
                ib+= 1
        
        if(len(tmp) < k):
            if(ib == len(y_score_1[i])):
                while(len(tmp) < k and it < len(y_score_2[i])):
                    if y_pred_2[i][it] not in tmp:
                        tmp.append(y_pred_2[i][it])
                        dic[y_pred_2[i][it]] = True
                    it+= 1
            else:
                while(len(tmp) < k and ib < len(y_score_1[i])):
                    if y_pred_1[i][ib] not in tmp:
                        tmp.append(y_pred_1[i][ib])
                        dic[y_pred_1[i][ib]] = True
                    ib+= 1
                    
        y_final.append(tmp)
            
    return  np.array(y_final)

def manage_row(row):
    """
        faire une fusion d'une ligne qui est constitué de la manière suivante:
        - la première moitié de la ligne contient les prédiction du premier model et la deuxième, celles du deuxième
        la ligne finale obtenue à une taille de len(row)//2 sachant que len(row) est paire car obtenue par la fusion de deux
        liste de même taille
    """
    # longueur commune des deux listes fusionnées pour obtenir row, et longueur de la liste obtenue après fusion
    k = len(row)//2 
    end1 = k//2 + 1*(k%2)
    dic = {}
    final = np.zeros(k)
    it = 0
    
    # on selectionne la moitié des premiers élements de la première partie
    for elt in row[: end1]:
        final[it] = elt
        dic[elt] = True
        it += 1
        
    # pour la deuxieme partie de la liste on vérifie d'abord si les élements ont déja été mis dans la liste finale
    # si c'est le cas on ne le met pas et on avance
    deb = k
    while it < k:
        if row[deb] not in dic:
            final[it] = row[deb]
            dic[row[deb]] = True
            it += 1
        deb += 1
        
            
    return final
    
def fusion_without_repetition (y_pred_1, y_pred_2):     
    y_fusion = np.concatenate((y_pred_1,y_pred_2), axis = 1)
    return np.apply_along_axis(lambda row : manage_row(row), 1, y_fusion)

def fusion(y_pred_1, y_pred_2, method = 'rep', y_score_1 = [], y_score_2 = []):
    """
        réalise la fusion en fonction de la méthode selectionnée
    """
    if method == 'basic':
        return fusion_basic(y_pred_1, y_pred_2)
    elif method == 'score':
        if (y_score_1 == [] or y_score_2 == []):
            print('score error')
            return
        else:
            return fusion_with_score(y_pred_1, y_score_1, y_pred_2, y_score_2)
    elif method == 'rep':
        return fusion_without_repetition (y_pred_1, y_pred_2)
    else:
        print('error method')
        return

def read_context_embedding(file):
    context_embedding = pd.read_csv(file, header=None)
    context_embedding = np.array(context_embedding)
    return context_embedding

def evaluate_model(model, questions, y_train, k = 1, batch_size = 100, iteration = 10, params = None, verbose = True):
    """
        evaluer le model sur plusieurs batchs générés aléatoirement
    """
    questions = pd.DataFrame(questions).iloc[:,0]
    n = len(questions)
    accuracy_list = []
    
    if k > model.k:
        print('the value of k is too large. the model can only predict : ' + str(model.k) + ' possibles documents')
        return
    
    for i in range(iteration):
        index_question = np.random.randint(n, size = batch_size)
        tmp_questions = questions.iloc[index_question] 
        y_true_label = np.array(y_train.iloc[index_question])
        context_id = np.array(model.context_id)
        
        if params == None:
            y_pred = model.predict(tmp_questions)
        elif 'method' in params:
            y_pred = model.predict(tmp_questions, method = params['method'])
        else:
            print('parameters error')
            return
        
        acc = []
        for j in range(model.k):
            acc.append(accuracy(y_true_label, y_pred, j+1))
        if verbose:
            print('batch '+str(i + 1) + ' : accuracy top '+str(k) +  ' = '+str(acc[0]))
        accuracy_list.append(acc)
    
    accuracy_list = np.array(accuracy_list)
    accuracy_list = accuracy_list.mean(axis = 0)
    
    return accuracy_list
    
    
def plot_accuracy(params, k):
    """
        tracer l'ensemble des accuracy contenues dans le dictionnaire param
    """
    plt.figure(figsize=(8,5))
    x_values = [i+1 for i in range(k)]
    for key in params:
        plt.plot(x_values, params[key][:k], label = key)
    plt.xticks(x_values)
    plt.xlabel('top -n')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


# In[ ]:




