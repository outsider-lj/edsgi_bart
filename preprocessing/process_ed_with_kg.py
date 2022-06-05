# Preprocess cornell movie dialogs dataset
import argparse
from pathlib import Path
import pandas as pd
project_dir = Path(__file__).resolve().parent
print(project_dir)
datasets_dir = project_dir.joinpath('../data/ed/')
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
from keybert import KeyBERT
from gensim.models import KeyedVectors
from nltk.corpus import sentiwordnet as swn
from itertools import chain
# nltk.download('averaged_perceptron_tagger')
stop= stopwords.words("english")+list(string.punctuation)
no_sense=list(['oh','ya','wow','yea','yeah','ah','wo','o','haha','ha','i','he','she','they','you','thats','that'])
stop_words=no_sense+stop
porter = PorterStemmer()
relation=[ 'Synonym', 'RelatedTo', 'MannerOf', 'EtymologicallyRelatedTo', 'FormOf',
        'IsA', 'DerivedFrom', 'EtymologicallyDerivedFrom', 'Causes', 'UsedFor', 'Antonym',
        'HasProperty', 'HasContext', 'MotivatedByGoal', 'AtLocation', 'SimilarTo', 'HasPrerequisite',
           'DistinctFrom', 'HasA', 'Desires', 'CapableOf', 'PartOf', 'CausesDesire', 'HasSubevent']
n = ['NN','NNP','NNPS','NNS','UH']
v = ['VB','VBD','VBG','VBN','VBP','VBZ']
a = ['JJ','JJR','JJS']
r = ['RB','RBR','RBS','RP','WRB']
MAX_CONCEPT=3
model = KeyBERT('distilbert-base-nli-mean-tokens')
def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
def normalization(data):
    _range=np.max(data)-np.min(data)
    return (data-np.min(data))/_range
def emotion_intensity(vad_tuple):
    '''
    Function to calculate emotion intensity (Eq. 1 in our paper)
    :param NRC: NRC_VAD vectors
    :param word: query word
    :return:
    '''
    v, a, d = vad_tuple
    a = a/2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468
def note_kg_word(utter,nrc,concept):
    emo_word=[]
    concept_word=[]
    concept_net= defaultdict(list)
    source = nltk.word_tokenize(utter)
    pos_tags = nltk.pos_tag(source)
    for word, pos in pos_tags:
        word= word.lower().strip()
        if pos in a + r and word not in stop_words:
            if word in nrc.keys():
                emo_word.append((word.lower(), pos))
        if pos in n + v and word not in stop_words:
            if word.lower() in concept.keys():
            # 保留词性为动词以及名词的实体
                concept_word.append((word.lower(), pos))
    return emo_word,concept_word

def note_dialog_concept_net(history,response,concept,dialog_concept,nrc,glove_dict):
    key_words = model.extract_keywords(history, keyphrase_ngram_range=(1, 1))
    key_words_list = [a[0] for a in key_words]
    source = nltk.word_tokenize(history)
    response=nltk.word_tokenize(response)
    pos_tags = nltk.pos_tag(source)
    for word, pos in pos_tags:
        word = word.lower().strip()
        if word not in stop_words and word in key_words_list:
            if word.lower() in concept.keys():
                if word in glove_dict.keys():
                    word_embedding = glove_dict[word]
                else:
                     word_embedding = np.zeros(300)
                     print(word, "not in glove dict")
                for j,triple in enumerate(concept[word]):
                    emo_scores=[]
                    concept_embeddings=[]
                    if triple[1]  in relation :
                        tail_entity = triple[0].split("_")
                        for i in tail_entity:
                            if i in nrc.keys():
                                emo_intensity = emotion_intensity(nrc[i])
                            else:
                                emo_intensity=0.0
                            emo_scores.append(emo_intensity)
                            if i in glove_dict.keys():
                                concept_embeddings.append(glove_dict[i])
                            else:
                                concept_embeddings.append(np.zeros(300))
                        concept_embedding = np.mean(concept_embeddings,axis=0)
                        emo_score = np.mean(emo_scores)
                        # word_embedding=normalization(word_embedding)
                        # concept_embedding=normalization(concept_embedding)
                # rs=np.dot(word_embedding.reshape(1,-1),concept_embedding.reshape(-1,1))
                        cos_score =get_cos_similar(word_embedding, concept_embedding)
                        score = cos_score+triple[2]+emo_score
                        concept[word][j]=(triple[0],triple[1],triple[2],score)
                concept_new=sorted(concept[word], key=lambda x: x[3], reverse=True)
                for item in concept_new:
                    if len(dialog_concept[word])>2:
                        break
                    dialog_concept[word].add(item)

                # for res_word in response:
                #     for triple in concept[word]:
                #         if str(res_word.lower())==str(triple[0]):
                #             print(word,triple)
                            # dialog_concept[word].add(triple)
                            # continue
                # if len(dialog_concept[word])<3:
                #     add_num=MAX_CONCEPT-len(dialog_concept[word])
                #     if len(concept[word])>add_num:
                #         for t in concept[word]:
                #             if t[1] in ["RelatedTo"]:
                #                 dialog_concept[word].add(t)
                #                 if len(dialog_concept[word]) == 3:
                #                     break
                #     else:
                #         for t in concept[word]:
                #             dialog_concept[word].add(t)
                # concept_word.append((word.lower(), pos))
        else:
            continue
    return dialog_concept
def is_topic_feedback(concept,utter,last_utter_type="emotion",is_subsentence=False):
    if utter[-1] == "?":
        return True
    if "sorry to hear"  in utter:
        return False
    if utter in freq_sent:
        return False
    emo_word, concept_word = note_kg_word(utter, nrc, concept)
    if concept_word==[] and emo_word!=[]:
        return False
    if emo_word==[] and concept_word!=[]:
        return  True
    elif len(concept_word) - len(emo_word) >1:
        return True
    if last_utter_type=="topic":
        if concept_word !=[]:
            return True
    else:
        return False

def is_emo_feedback(concept,utter,last_utter_type="emotion",is_subsentence=False):
    if "sorry" in utter:
        return True
    if "love" in utter:
        return True
    if "congratulation" in utter:
        return True
    if utter in freq_sent:
        return True
    emo_word, concept_word = note_kg_word(utter, nrc, concept)
    if emo_word != [] and concept_word == []:
        return  True
    if emo_word ==[] and concept_word != []:
        return  False
    elif len(emo_word) - len(concept_word) > 1:
        return True
    else:
        return False
def clean(sentence,word_pairs1,word_pairs2):
    for k, v in word_pairs1.items():
        sentence = sentence.replace(k,v)
    for k, v in word_pairs2.items():
        sentence = sentence.replace(k,v)
    return sentence

def get_one_instance(num,emotions,situation,utterance,emotion,history,nrc,concept,freq_sent,glove_dict):
    dialog_concept=defaultdict(set)
    # ctrl=set()
    instance = {}
    instance["emotion"] = emotions
    instance["situation"] = [situation]
    # instance["emotions"] = [emotion]
    instance["reply"]=[utterance]
    instance["history"]=history
    dialog_concept=note_dialog_concept_net(" ".join(history).strip(),utterance,concept,dialog_concept,nrc,glove_dict)
    instance["concept_net"]=dialog_concept
    emo_sent=[]
    topic_sent=[]
    #将回复差分成两个句子
    sep_utterance = re.split(r"([.!?~;]*)",utterance.strip())
    sep_utterance.append("")
    utterance_sent=["".join(i) for i in zip(sep_utterance[0::2], sep_utterance[1::2])]#切分后保留标点符号
    # print(utterance_sen)
    if utterance_sent[-1]=='':
        utterance_sent=utterance_sent[:-1]
    type="emotion"
    is_subsentence = False
    for utter in utterance_sent:
        if is_topic_feedback(concept,utter.lower().strip(),type,is_subsentence)==True:
            is_topic=True
        else:
            is_topic=False
        if is_emo_feedback(concept,utter.lower().strip(),type,is_subsentence)==True:
            is_emo=True
        else:
            is_emo=False
        if is_topic==True and is_emo==False:
            topic_sent.append(utter.strip())
            type = "topic"
        elif is_topic==False and is_emo==True:
            emo_sent.append(utter.strip())
            type = "emotion"
        else :
            if utter[-1]=="?":
                topic_sent.append(utter.strip())
                type = "topic"
            elif len(nltk.word_tokenize(utter.strip()))>10:
                # print(utter)
                topic_sent.append(utter.strip())
                type = "topic"
            # elif utter[-1]=="!":
            #     emo_sent.append(utter.strip())
            #     type = "emotion"
            # elif len(nltk.word_tokenize(utter.strip())) < 4:
            #     emo_sent.append(utter.strip())
            #     type = "emotion"
            else:
                emo_sent.append(utter.strip())
                type = "emotion"
        if utter.strip()[-1]==",":
            is_subsentence==True
        else:
            is_subsentence==False
    # instance["ctrl"]=list(ctrl)
    instance["reply_first"]=emo_sent
    instance["reply_second"]=topic_sent

    #加一个长度关系的判断
    return instance#,no_sense

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

if __name__ == '__main__':
    word_pairs1 = {" thats ": " that's ", " dont ": " don't ", " doesnt ": " doesn't ", " didnt ": " didn't ",
                   " youd ": " you'd ",
                   " youre ": " you're ", " youll ": " you'll ", " im ": " i'm ", " theyre ": " they're ",
                   " whats ": "what's", " couldnt ": " couldn't ", " souldnt ": " souldn't ", " ive ": " i've ",
                   " cant ": " can't ", " arent ": " aren't ", " isnt ": " isn't ", " wasnt ": " wasn't ",
                   " werent ": " weren't ", " wont ": " won't ", " theres ": " there's ", " therere ": " there're "}

    word_pairs2 = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}

    # glove_dict=np.load('./data/worddict.npy', allow_pickle=True)
    # print('Loaded the word dict!')
    # wordsList = wordsList.tolist()  # Originally loaded as numpy array
    # wordVectors = np.load('./data/wordVectors.npy')
    glove_dict = pickle.load(open("../data/glove/worddict.pkl", "rb"))
    nrc = pickle.load(open("../data/kgs/NRC_VAD.pkl", "rb"))
    concept = pickle.load(open("../data/kgs/filtered_conceptnet.pkl", "rb"))
    freq_sent_file = open("../data/ed/ED_freq_sent.txt", "r")
    freq_sent = freq_sent_file.readlines()
    for i, s in enumerate(freq_sent):
        freq_sent[i] = freq_sent[i].strip()
    json_file = open('../data/ed/ed_with_kg_clean.json', 'w+', encoding='utf-8')
    # no_sense_file=open('ED/no_sense.txt', 'w+', encoding='utf-8')
    all_no_sense_sent=[]
    datasets = {"train":defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
    for type in ["train","valid","test"]:
        k=0
        j=0
        one=0
        more=0
        fileName="../data/ed/"+type+".csv"
        dataframe = pd.read_csv(open(fileName), encoding="utf-8", delimiter="\t")
        for i in range(dataframe.size):
            line = dataframe.loc[i].values[0].split(",")
            num=line[1]
            emotion = "<"+line[2]+">"
            situation = line[3].replace("_comma_", ",").lower()
            utterance =line[5].replace("_comma_", ",").lower()
            utterance = clean(utterance,word_pairs1,word_pairs2)
            situation=clean(situation,word_pairs1,word_pairs2)
            if int(num) == 1:
                if i==0:
                    conv={"personality":[], "utterances": []}
                else:
                    if k==0:
                        datasets[type]=[conv]
                        j = 0
                        k += 1
                    else:
                        datasets[type].append(conv)
                        j=0
                    conv={"personality": [], "utterances": []}
                emotions=[emotion]
                history = [utterance]
            elif int(num) % 2==1:
                history.append(utterance)
                emotions.append(emotion)
            else:
                instance={}
                e=emotions[:len(emotions)]
                h=history[:len(history)]
                instance=get_one_instance(num,e,situation,utterance,emotion,h,nrc,concept,freq_sent,glove_dict)
                all_no_sense_sent.append(no_sense)
                history.append(utterance)
                emotions.append(emotion)
                if j==0:
                    conv["utterances"]=[instance]
                    j+=1
                else:
                    conv["utterances"].append(instance)
                if i==dataframe.size-1:
                    datasets[type].append(conv)
        # print(one,more)
    json_file.write(json.dumps(datasets, default=set_default,ensure_ascii=False))
    json_file.close()

