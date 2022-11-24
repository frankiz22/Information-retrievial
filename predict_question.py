#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import argparse
import pandas as pd
from utils import read_context_embedding
from models import Combined_model
import sys
import timeit




def main():
    print("start...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs=1, required = True)
    parser.add_argument('--question', type=str, nargs=1, required = True)
    args, unknown = parser.parse_known_args()
    file_name = args.dataset[0]
    question = args.question[0]
    
      
    
    
    if file_name == "train_set":
        print("loading datasets...", flush= True)
        X_question = pd.read_csv('Data/train_data/train_questions.csv')
        X_context = pd.read_csv('Data/train_data/train_contexts.csv')
        context_embedding = read_context_embedding('Data/train_data/train_context_embbeding_with_bert.csv')
        
    elif file_name == "val_set":
        print("loading datasets...", flush = True)
        X_question = pd.read_csv('Data/val_data/val_questions.csv')
        X_context = pd.read_csv('Data/val_data/val_contexts.csv')
        context_embedding = read_context_embedding('Data/val_data/val_context_embbeding_with_bert.csv')
    else:
        print("file name error")
        return
    
    print("loading model ...", flush = True)
    
    model1_params = {'name': 'transformer',
                 'vectorizer': 'default',
                 'context_embedding': context_embedding}

    model2_params = {'name': 'tfidf',
                 'vectorizer': 'default'}
    
    context_id = X_context.index
    model = Combined_model(context_id,X_context['text_context'], model1_params, model2_params, 1)
    model.fit()
    print("predicting question ...", flush = True)
    
    start = timeit.default_timer()
    y = model.predict([question])
    stop = timeit.default_timer()
    
    print("\nle context est : "+ X_context.loc[y[0][0]]["text_context"])
    print("\ntime for question prediction : " +  str(stop - start))
    
    

    
if __name__ == "__main__":
    main()


# In[ ]:




