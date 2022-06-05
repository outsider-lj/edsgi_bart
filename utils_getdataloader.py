from torch.utils.data import DataLoader, TensorDataset
import torch
import pickle
import logging
import os
from collections import defaultdict
import numpy as np
import json
import nltk
BOS_TOKEN_ID = 0
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
EMO_PAD_TOKEN_ID=8
MAX_LENGTH = [149,15, 31, 39]
n = ['NN','NNP','NNPS','NNS','UH']#5
v = ['VB','VBD','VBG','VBN','VBP','VBZ']#6
a = ['JJ','JJR','JJS']#3
r = ['RB','RBR','RBS','RP','WRB']#5
PAD_TOKEN="<pad>"
EMO_PAD="other"
MASK=[0,1,2]
special_tags=["<s>","</s>","<pad>","<mask>","<unk>","<sep>","<cls>"]
no_sense=list(['oh','ya','wow','yea','yeah','ah','wo','o','haha','ha','i','he','she','they','you','thats','that'])
speaker=["<speaker1>","<speaker2>"]
NRC_CLASS={}
nrc=["<anger>","<anticipation>","<disgust>","<fear>","<joy>","<sadness>","<surprise>","<trust>","<other>"]
for i,k in enumerate(nrc):
    NRC_CLASS[k]=i
PAD_EMO_ID=NRC_CLASS["<other>"]
logger = logging.getLogger(__file__)

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

def add_knowledge_with_vm(sent_batch,concept,nrc, tokenizer, add_pad=True, max_length=128):
    """
    input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
    return: know_sent_batch - list of sentences with entites embedding
            position_batch - list of position index of each character.
            visible_matrix_batch - list of visible matrixs
            seg_batch - list of segment tags
    """
    split_sent =["<s>"]
    for sent in sent_batch:
        split_sent+= tokenizer.tokenize(sent)+ ["</s>"] #或者用tokenizer.cut
    # split_sent_batch = [["<s>"] + tokenizer.tokenize(sent) + ["</s>"] for sent in sent_batch]
    split_sent_batch=[split_sent]
    know_sent_batch = []
    knowledge_confidence_batch=[]
    emo_cls_batch=[]
    position_batch = []
    visible_matrix_batch = []
    seg_batch = []
    token_type_batch=[]
    for i, split_sent in enumerate(split_sent_batch):
        sent_tree = []
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1
        abs_idx = -1
        abs_idx_src = []
        for token in split_sent:
            if token=='':
                continue
            # if token in special_tags:
            #     sep_token=[token]
            # else:
            #     sep_token=tokenizer.tokenize(token)
            # sep_token=[i for i in sep_token if i !="_"]
            if token in concept.keys():
                entities=list([" "+t[0] for t in concept[token]])
            else:
                entities=[]
            sent_tree.append((token, entities))
            if token in special_tags:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
            else:
                token_pos_idx = [pos_idx + i for i in range(1, 2)]#len(sep_token) + 1
                token_abs_idx = [abs_idx + i for i in range(1, 2)]
            abs_idx = token_abs_idx[-1]

            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                sep_ent = tokenizer.tokenize(ent)
                sep_ent=[i for i in sep_ent if i!="_"]
                ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(sep_ent) + 1)]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + i for i in range(1, len(sep_ent) + 1)]
                abs_idx = ent_abs_idx[-1]
                entities_abs_idx.append(ent_abs_idx)

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        # Get know_sent and pos 在完成语义树与位置数后，对词与进行token分词，更新输入的索引
        know_sent = []
        pos = []
        emo_cls=[]
        knowledge_confidence=[]
        seg = []#插入知识的位置
        token_type=[]
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if word in special_tags:
                know_sent += [word]
                seg += [0]
                emo_cls+=[0.0]
                knowledge_confidence+=[0.0]
            else:
                # add_word = tokenizer.tokenize(word)
                # add_word = [i for i in add_word if i != "_"]
                know_sent += [word]
                seg += [0] #* len(add_word)
                if len(sent_tree[i][1])>0:
                    knowledge_confidence+=[1.0]
                else:
                    knowledge_confidence+=[0.0]
                if "Ġ" in word:
                    word = word.replace("Ġ", '')

                if word in nrc.keys() and nltk.pos_tag([word])[0][1] in n+v+a+r and word not in no_sense:
                    emo =emotion_intensity(nrc[word])
                    emo_cls+=[emo]#*len(add_word)
                else:
                    # emo=emotion_intensity((0.5,0.5,0))
                    emo_cls+=[0]#*len(add_word)
            pos += pos_idx_tree[i][0]
            if len(sent_tree[i][1])>0:
                klg=concept["Ġ"+word]
            for j in range(len(sent_tree[i][1])):
                for item in klg:
                    if item[0]==sent_tree[i][1][j].strip():
                            word_confidence=item[2]
                            break
                add_word = tokenizer.tokenize(sent_tree[i][1][j])
                add_word = [i for i in add_word if i != "_"]
                know_sent += add_word
                knowledge_confidence+=[word_confidence]*len(add_word)
                if sent_tree[i][1][j] in nrc.keys():
                    emo =emotion_intensity(nrc[sent_tree[i][1][j]])
                    emo_cls+=[emo]*len(add_word)
                else:
                    # emo=emotion_intensity((0.5,0.5,0))
                    emo_cls+=[0]*len(add_word)
                seg += [1] * len(add_word)
                pos += list(pos_idx_tree[i][1][j])#* len(add_word)

        token_num = len(know_sent)
        emo_num=len(emo_cls)
        t=1
        for word in know_sent:
            if t == 1:
                token_type.append(speaker[0])
            else:
                token_type.append(speaker[1])
            if word =="</s>":
                t = -t
        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                if visible_abs_idx[-1] >=token_num:
                    print(sent_batch)
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [PAD_TOKEN] * pad_num
            emo_cls += [0] * pad_num
            seg += [PAD_TOKEN_ID] * pad_num
            pos += [max_length - 1] * pad_num
            knowledge_confidence+=[0]*pad_num
            token_type +=[PAD_TOKEN] * pad_num
            visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            emo_cls=emo_cls[:max_length]
            knowledge_confidence=knowledge_confidence[:max_length]
            token_type=token_type[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]

        know_sent_batch.append(know_sent)
        emo_cls_batch.append(emo_cls)
        position_batch.append(pos)
        visible_matrix_batch.append(visible_matrix)
        seg_batch.append(seg)
        knowledge_confidence_batch.append(knowledge_confidence)

        token_type_batch.append(token_type)

    return know_sent_batch,emo_cls_batch,knowledge_confidence_batch, position_batch,\
           visible_matrix_batch, seg_batch,token_type_batch



def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_en_l = max(len(x) for x in dataset["situation_ids"])
    max_de_l1 = max(len(x) for x in dataset["decoder_input_ids_first"])
    max_de_l2 = max(len(x) for x in dataset["decoder_input_ids_second"])
    max_de_l = max(len(x) for x in dataset["decoder_input_ids_final"])
    # dataset["input_ids"] = [x + [padding] * (max_en_l - len(x)) for x in dataset["input_ids"]]
    dataset["decoder_input_ids_first"] = [x + [padding] * (max_de_l1 - len(x)) for x in
                                          dataset["decoder_input_ids_first"] ]
    dataset["decoder_input_ids_second"] = [x + [padding] * (max_de_l2 - len(x)) for x in
                                           dataset["decoder_input_ids_second"]]
    dataset["decoder_input_ids_final"] = [x + [padding] * (max_de_l - len(x)) for x in
                                          dataset["decoder_input_ids_final"]]
    dataset["lm_labels_first"] = [x + [padding] * (max_de_l1 - len(x)) for x in dataset["lm_labels_first"]]
    dataset["lm_labels_second"] = [x + [padding] * (max_de_l2 - len(x)) for x in dataset["lm_labels_second"]]
    # dataset["lm_kg_first"] = [x + [padding] * (max_de_l1 - len(x)) for x in dataset["lm_kg_first"]]
    # dataset["lm_kg_second"] = [x + [padding] * (max_de_l2 - len(x)) for x in dataset["lm_kg_second"]]
    dataset["lm_labels_final"] = [x + [padding] * (max_de_l -len(x)) for x in dataset["lm_labels_final"]]
    dataset["situation_ids"] = [x + [padding] * (max_en_l - len(x)) for x in dataset["situation_ids"]]
    # dataset["emo_input1_ids"] = [x + [EMO_PAD_TOKEN_ID] * (max_de_l1 - len(x)) for x in dataset["emo_input1_ids"]]
    # dataset["emo_input2_ids"] = [x + [EMO_PAD_TOKEN_ID] * (max_de_l2 - len(x)) for x in dataset["emo_input2_ids"]]
    # dataset["emo_inputfinal_ids"] = [x + [EMO_PAD_TOKEN_ID] * (max_de_l - len(x)) for x in dataset["emo_inputfinal_ids"]]
    # dataset["emo_token_ids"] = [0 for x in dataset["emo_token_ids"]]
    return dataset


def build_input_from_segments(tokenizer,situation,reply1,reply1_kg,reply2,reply2_kg,reply,emotion, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos = BOS_TOKEN_ID
    eos = EOS_TOKEN_ID
    # cls=tokenizer.convert_tokens_to_ids("<cls>")
    reply_ctrl=[]
    instance = {}
    #对上下文、情感反馈、话题回复、最后回复分别处理
    if len(situation) > MAX_LENGTH[0]-1:
        situation = situation[:MAX_LENGTH[0]-1]
    # situation = [cls] + situation + [eos]
    if reply1==[]:
        # reply_ctrl.append(no_es_id)
        lm_labels1=decoder1=lm1_kg=[PAD_TOKEN_ID]
        # emo_input1 =[PAD_EMO_ID]
    else:
        # reply_ctrl.append(es_id)
        if len(reply1) > MAX_LENGTH[1]:
            reply1 = reply1[:MAX_LENGTH[1]]
            reply1_kg = reply1_kg[:MAX_LENGTH[1]]
            # reply1_emo = reply1_emo[:MAX_LENGTH[1]]
        lm_labels1 =reply1 + ([eos] if with_eos else [])
        decoder1=[bos]+ reply1
        lm1_kg=reply1_kg+[PAD_TOKEN_ID]
        # emo_input1=[PAD_EMO_ID]+reply1_emo+[PAD_EMO_ID]
    if reply2==[]:
        # reply_ctrl.append(no_ts_id)
        lm_labels2=decoder2=lm2_kg=[PAD_TOKEN_ID]
        # emo_input2 =[PAD_EMO_ID]
    else:
        # reply_ctrl.append(ts_id)
        if len(reply2) > MAX_LENGTH[2]:
            reply2 = reply2[:MAX_LENGTH[2]]
            reply2_kg = reply2_kg[:MAX_LENGTH[2]]
            # reply2_emo = reply2_emo[:MAX_LENGTH[2]]
        lm_labels2=reply2 + ([eos] if with_eos else [])
        decoder2=[bos]+reply2
        lm2_kg =reply2_kg + [PAD_TOKEN_ID]
        # emo_input2 = [PAD_EMO_ID] + reply2_emo + [PAD_EMO_ID]
    if len(reply) > MAX_LENGTH[3]:
        reply = reply[:MAX_LENGTH[3]]
        # reply_emo = reply_emo[:MAX_LENGTH[2]]
    #将处理好的句子连接到对应的key
    # instance["input_ids"] = history#[bos] +history + ([eos] if with_eos else [])
    # instance["history_pos"]=history_pos
    # instance["vm"]=vm
    instance["decoder_input_ids_first"] = decoder1+ [PAD_TOKEN_ID] * (MAX_LENGTH[1] + 1 - len(decoder1))
    instance["decoder_input_ids_second"] = decoder2+[PAD_TOKEN_ID] * (MAX_LENGTH[2] + 1 - len(decoder2))
    instance["decoder_input_ids_final"]=[bos]+reply+[PAD_TOKEN_ID] * (MAX_LENGTH[3] - len(reply))
    instance["lm_labels_first"] = lm_labels1+[PAD_TOKEN_ID] * (MAX_LENGTH[1] + 1 - len(lm_labels1))
    instance["lm_labels_second"] = lm_labels2+[PAD_TOKEN_ID] * (MAX_LENGTH[2] + 1 - len(lm_labels2))
    instance["lm_labels_final"] = reply + ([eos] if with_eos else [])+[PAD_TOKEN_ID] * (MAX_LENGTH[3] - len(reply))
    # instance["lm_kg_first"] = lm1_kg#+[PAD_TOKEN_ID] * (MAX_LENGTH[1] + 1 - len(lm1_kg))
    # instance["lm_kg_second"] = lm2_kg#+[PAD_TOKEN_ID] * (MAX_LENGTH[2] + 1 - len(lm2_kg))
    instance["emo_label"] = emotion#19为词性的特殊标记数量
    # instance["situation_ids"] = situation + [PAD_TOKEN_ID] * (MAX_LENGTH[0] + 1 - len(situation))
    return instance

def reply_word_to_ids(kg_words,tokenizer,reply):
    # kg_words=["Ġ"+k for k in kg_words]
    kg=[]
    reply_token_ids=[]
    emo_input_ids=[]
    if reply==[]:
        # reply_token_ids.append(PAD_TOKEN_ID)
        # kg.append(PAD_TOKEN_ID)
        return reply_token_ids,kg
    else :
        for s in reply:
            s=tokenizer.tokenize(s.strip())
            for w in s:


                # tokens=tokenizer.tokenize(w)
                ids=tokenizer.convert_tokens_to_ids(w)# for t in tokens]
                reply_token_ids+=[ids]
                if kg_words is not None:
                    if "Ġ" in w:
                       w=w.replace("Ġ",'')
                    if len(w.strip())<1:
                        kg += [PAD_TOKEN_ID]
                    elif nltk.pos_tag([w.strip()]) not in n + r + a + v or w in no_sense:
                        kg += [PAD_TOKEN_ID]
                    elif str(w.strip()) in list(kg_words):
                        kg+=[ids]
                        # print(w)
                    else:
                        kg+=[PAD_TOKEN_ID]#*len(ids)
                # if nrc_dict is not None:
                #     if w in nrc_dict.keys():
                #         emo_input_ids+=[NRC_CLASS["<"+nrc_dict[w][0]+">"]]#*len(ids)
                #     else:
                #         emo_input_ids+=[NRC_CLASS["<other>"]]#*len(ids)
        # reply=[tokenizer.encode(s) for s in reply]
    # reply=[BOS_TOKEN_ID]+reply+[EOS_TOKEN_ID]
    # kg=[BOS_TOKEN_ID]+reply+[EOS_TOKEN_ID]
    return reply_token_ids,kg
def select_dict( concept ):
    if len(concept) <=3:
        num=3
    elif len(concept) <= 5:
        num = 2
    else:
        num = 1
    if len(concept) > 10:
        s = []
        for k, items in concept.items():
            for item in items:
                if len(concept) > 10:
                    s.append(item[3])
        s = sorted(s)
        mins = s[9]
    else:
        mins = 0.0
    bart_concept = {}
    for k, items in concept.items():
        k = "Ġ" + k
        # for v in item:
        #     v[2]=(v[2]-min)/max
        items = sorted(items, key=lambda x: x[3], reverse=True)
        if len(concept) <= 10:
            bart_concept[k] = items[:num]
        else:
            for item in items:
                if item[3] > mins:
                    bart_concept[k].append(item)
                else:
                    continue
    return bart_concept
def get_data_loaders(config, tokenizer,emo_labels_dict,model_inputs):
    # pad=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
    """ Prepare the dataset for training and evaluation """
    nrc = pickle.load(open("data/kgs/NRC_VAD.pkl", "rb"))
    # nrc_words=nrc.keys()
    pad = PAD_TOKEN_ID
    dataset_cache=config.dataset_cache
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        datasets = torch.load(dataset_cache)
    else:
        logger.info("Tokenize and encode the dataset")
        with open(config.dataset_path, "r", encoding="utf-8") as f:
            chat_dataset = json.loads(f.read())
        logger.info("Build inputs and labels")
        datasets = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
        # gpu_max_length = 310 #this depends on the gpu memory size, using bigger gpu memory you can increase this to include longer inputs
        for dataset_name, dataset in chat_dataset.items():
            for dialog in dataset:
                for utterance in dialog["utterances"]:
                    if utterance == []:
                        continue
                    concept = utterance["concept_net"]
                    bart_concept=select_dict(concept)
                    concept_words = [t[0] for k,v in bart_concept.items() for t in v]
                    history = utterance["history"][-(2 * config.max_history + 1):]
                    # history=[" </s> ".join(history)]
                    history_tokens,emo_intensity,knowledge_confidence, history_pos, vm, tag,token_type = add_knowledge_with_vm(history, bart_concept,nrc,
                                                                                          tokenizer, add_pad=True,
                                                                                          max_length=MAX_LENGTH[0]+1)
                    history_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in history_tokens[0]]
                    token_type_ids=[tokenizer.convert_tokens_to_ids(t) for t in token_type[0]]
                    # emo_input_ids = [NRC_CLASS["<"+t+">"] for t in emo_input_tokens[0]]
                    tag=[1 if i==0 else 0 for i in tag[0]]
                    reply1,reply1_kg_label = reply_word_to_ids(nrc,tokenizer, utterance["reply_first"])
                    reply2,reply2_kg_label = reply_word_to_ids(concept_words,tokenizer, utterance["reply_second"])
                    reply,reply_kg_label =reply_word_to_ids(None,tokenizer,utterance["reply"]) #[tokenizer.encode(s) for s in utterance["reply"]]
                    emotion = emo_labels_dict[utterance["emotion"][0]]
                    situation = [tokenizer.encode(s) for s in utterance["situation"]][0]
                    # lm_labels = bool(j == num_candidates-1) #the true label is always the last one in list of candidates
                    instance= build_input_from_segments(tokenizer,situation,reply1,reply1_kg_label,reply2,reply2_kg_label,reply, emotion)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                    datasets[dataset_name]["input_ids"].append(history_token_ids)
                    datasets[dataset_name]["knowledge_confidence"].append(knowledge_confidence)# [bos] +history + ([eos] if with_eos else [])
                    datasets[dataset_name]["position_ids"].append(history_pos[0])
                    datasets[dataset_name]["vm"].append(vm[0])
                    datasets[dataset_name]["emo_intensity"].append(emo_intensity)
                    datasets[dataset_name]["tag"].append(tag)
                    datasets[dataset_name]["token_type_ids"].append(token_type_ids)

        if dataset_cache:
            logger.info("Saving caches...")
            torch.save(datasets, dataset_cache)
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": [], "test": []}
    for dataset_name, dataset in datasets.items():
        # dataset = pad_dataset(dataset, padding=pad)
        for input_name in model_inputs:
            if input_name == "attention_mask":
                for inputs in dataset["input_ids"]:
                    att_mask = [0 if ids == pad else 1 for ids in inputs]
                    dataset["attention_mask"].append(att_mask)
            if input_name == "situation_mask":
                for inputs in dataset["situation_ids"]:
                    att_mask = [0 if ids == pad else 1 for ids in inputs]
                    dataset["situation_mask"].append(att_mask)
            if input_name == "decoder_attention_mask_first":
                for inputs in dataset["decoder_input_ids_first"]:
                    att_mask = [0 if ids == pad else 1 for ids in inputs]
                    dataset["decoder_attention_mask_first"].append(att_mask)
            if input_name == "decoder_attention_mask_second":
                for inputs in dataset["decoder_input_ids_second"]:
                    att_mask = [0 if ids == pad else 1 for ids in inputs]
                    dataset["decoder_attention_mask_second"].append(att_mask)
            if input_name == "decoder_attention_mask_final":
                for inputs in dataset["decoder_input_ids_final"]:
                    att_mask = [0 if ids == pad else 1 for ids in inputs]
                    dataset["decoder_attention_mask_final"].append(att_mask)
            tensor = torch.tensor(dataset[input_name])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["test"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if config.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if config.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.train_batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=config.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler