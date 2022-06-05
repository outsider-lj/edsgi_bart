import argparse
from pathlib import Path
import pandas as pd
project_dir = Path(__file__).resolve().parent
print(project_dir)
datasets_dir = project_dir.joinpath('ED/')
import csv
import json
from collections import defaultdict
import re
import pickle
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from nltk.stem import PorterStemmer
stop= stopwords.words("english")+list(string.punctuation)
no_sense=list(['oh','ya','wow','yea','yeah','ah','wo','o','haha','ha'])
stop_words=no_sense+stop
porter = PorterStemmer()
relation=['RelatedTo', 'MannerOf', 'EtymologicallyRelatedTo',
        'IsA', 'Causes', 'UsedFor','HasProperty', 'HasContext', 'MotivatedByGoal', 'SimilarTo',
           'HasPrerequisite', 'HasA', 'Desires', 'CapableOf', 'PartOf', 'CausesDesire', 'HasSubevent']# 'Synonym',
excluded_relation_list=["Antonym","ExternalURL",
"NotDesires","NotHasProperty","NotCapableOf","dbpedia","DistinctFrom",
"EtymologicallyDerivedFrom","EtymologicallyRelatedTo","SymbolOf","FormOf",
"AtLocation,DerivedFrom","SymbolOf"]
N = ['NN','NNP','NNPS','NNS','UH']
V = ['VB','VBD','VBG','VBN','VBP','VBZ']
A = ['JJ','JJR','JJS']
R = ['RB','RBR','RBS','RP','WRB']

def normalization(data):
    _range=np.max(data)-np.min(data)
    return (data-np.min(data))/_range
def is_all_eng(strs):
    import string
    for i in strs:
        if i not in string.ascii_lowercase+string.ascii_uppercase:
            return False
    return True

def get_emotion_intensity(NRC, word):
    if word not in NRC:
        word = porter.stem(word)
        if word not in NRC:
            return 0.01
    v, a = NRC[word]
    # v, a, d = NRC[word]
    # a = a/2
    return a#(np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468

# remove cases where the same concept has multiple weights
def remove_KB_duplicates(conceptnet):
    filtered_conceptnet = {}
    for k in conceptnet:
        filtered_conceptnet[k] = set()
        concepts = set()
        filtered_concepts = sorted(conceptnet[k], key=lambda x: x[2], reverse=True)
        for c,w in filtered_concepts:
            if c not in concepts:
                filtered_conceptnet[k].add((c, w))
                concepts.add(c)
    return filtered_conceptnet


def filtered_knowledge(word,concept,nrc,glove_dict):
    emo_scores = []

    filtered_knowledge = set()
    filtered_conceptnet = set()
    concepts = set()
    filtered_concepts = sorted(concept, key=lambda x: x[2], reverse=True)
    max=1.0
    min=10.0
    for c,r,w in filtered_concepts:
        # if w > max:
        #     max = w
        # if w < min:
        #     min = w
        if c not in concepts:#将重复的尾结点去除
            filtered_conceptnet.add((c,r,w))
            concepts.add(c)
    # max+=min+0.1
    if word in glove_dict.keys():
        word_embedding = glove_dict[word]
    else:
        word_embedding = np.zeros(300)
        print(word, "not in glove dict")
    for triple in iter(filtered_conceptnet):
        if triple[1] in relation and triple[2] > 1 : #and triple[0] in nrc.keys():
            concept_embeddings = []
            tail_entity = triple[0].split("_")
            if is_all_eng(tail_entity[0]) is False:
                continue
            pos=nltk.pos_tag(tail_entity)
            if pos[0][1] not in N+V+A+R:
                continue
            # for i in tail_entity:
                # emo_intensity = get_emotion_intensity(nrc, i)
                # emo_scores.append(emo_intensity)
                # if i in glove_dict.keys():
                #     concept_embeddings.append(glove_dict[i])
            # if len(concept_embeddings)>1:
            #     concept_embedding = np.mean(concept_embeddings,axis=0)
            # elif len(concept_embeddings)==1:
            #     concept_embedding=concept_embeddings[0]
            # else:
            #     concept_embedding=np.zeros(300)
            confidence_score = (triple[2] - min) / (max-min)
            # emo_score = np.mean(emo_scores)
            # word_embedding=normalization(word_embedding)
            # concept_embedding=normalization(concept_embedding)
            # rs=np.dot(word_embedding.reshape(1,-1),concept_embedding.reshape(-1,1))
            # cos_score =np.mean(np.cos(word_embedding, concept_embedding))
            # score = (confidence_score + cos_score)/2
            if  confidence_score>0.1:
                filtered_knowledge.add((triple[0], triple[1], confidence_score))

        else:
            continue
        # filtered_knowledge=remove_KB_duplicates(filtered_knowledge)
    return filtered_knowledge
def select_knowldege(triple,nrc):
    #输入的为（尾实体，关系，置信度）的三元组
    if triple[1] in relation and triple[2] > 1:
        tail_entity=triple[0].split("_")
        for i in tail_entity:
            if i in nrc.keys():
                return True
            else:
                return False
    else:
        return False




if __name__ == '__main__':
    conceptnet=defaultdict(list)
    glove_dict=pickle.load(open("../data/glove/worddict.pkl","rb"))
    print('Loaded the word dict!')
    # wordsList = wordsList.tolist()  # Originally loaded as numpy array
    # wordVectors = np.load('./data/wordVectors.npy')
    nrc = pickle.load(open("../data/kgs/NRC.pkl", "rb"))
    concept = pickle.load(open("../data/kgs/concept.pkl", "rb"))
    print(max,min)
    for w,c in concept.items():
        # filtered_conceptnet = remove_KB_duplicates()
        filtered_conceptnet=filtered_knowledge(w,c,nrc,glove_dict)
        if len(filtered_conceptnet) <1:
            continue
        filtered_conceptnet = sorted(filtered_conceptnet, key=lambda x: x[2], reverse=True)
        conceptnet[w]=filtered_conceptnet
    pickle.dump(conceptnet,open("../data/kgs/filtered_conceptnet.pkl","wb"))
    print("saved filter_conceptnet!")
