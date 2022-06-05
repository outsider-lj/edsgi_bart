# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import numpy as np
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from config import InteractConfig
from bart import BartTokenizer
from bart import StepBartForDialogueGeneration
import  eval
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from utils import download_pretrained_model,  _f1_score
from utils_getdataloader import add_knowledge_with_vm
import nltk
import pickle
PAD_TOKEN=1
BOS_TOKEN=0
EOS_TOKEN=2
# SPECIAL_TOKENS = ["<speaker1>", "<speaker2>"]
MAX_LENGTH = [148, 14, 30, 38]
n = ['NN', 'NNP', 'NNPS', 'NNS', 'UH']  # 5
v = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']  # 6
a = ['JJ', 'JJR', 'JJS']  # 3
r = ['RB', 'RBR', 'RBS', 'RP', 'WRB']  # 5
EMO_LABELS=["<joyful>", "<excited>", "<proud>", "<grateful>", "<hopeful>", "<content>",
                                      "<prepared>", "<anticipating>", "<confident>",
                                      "<sentimental>", "<nostalgic>", "<trusting>", "<faithful>", "<caring>",
                                      "<terrified>", "<afraid>", "<anxious>", "<apprehensive>",
                                      "<lonely>", "<embarrassed>", "<ashamed>", "<guilty>", "<sad>", "<disappointed>",
                                      "<devastated>","<angry>", "<annoyed>", "<disgusted>", "<furious>", "<jealous>",
                                      "<impressed>", "<surprised>"]
NRC_CLASS={}
nrc=["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","other"]
for i,k in enumerate(nrc):
    NRC_CLASS[k]=i
PAD_EMO_ID=NRC_CLASS["other"]

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

def  build_emo_input(nrc,tokenizer,sent):
    reply_ids=[]
    emo_input_ids=[]
    # sep_sent=nltk.word_tokenize(sent)
    for word in sent:
        id = tokenizer.convert_tokens_to_ids(word)
        reply_ids.append(id)
        if "Ġ" in word:
            word=word.replace("Ġ",'')
        if word in nrc.keys():
            emo_input_ids+=[NRC_CLASS[nrc[word][0]]]#*len(ids)
        else:
            emo_input_ids+=[NRC_CLASS["other"]]#*len(ids)
    return reply_ids

def build_input_from_segments(nrc_dict,reply1, reply2,emotion, tokenizer, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos = BOS_TOKEN
    eos = EOS_TOKEN
    instance = {}
    if len(reply1) > MAX_LENGTH[1]:
        reply1 = reply1[:MAX_LENGTH[1]]
        # emo_input1_ids = emo_input1_ids[:MAX_LENGTH[1]]
    if len(reply2) > MAX_LENGTH[2]:
        reply2 = reply2[:MAX_LENGTH[2]]
        # emo_input2_ids = emo_input2_ids[:MAX_LENGTH[2]]
    #对上下文、情感反馈、话题回复、最后回复分别处理
    decoder_input_ids1 =[bos]+ reply1 + ([eos] if with_eos else [])
    decoder_input_ids2 = [bos]+ reply2 + ([eos] if with_eos else [])
    #将处理好的句子连接到对应的key
    instance["decoder_input_ids_first"] = decoder_input_ids1
    instance["decoder_input_ids_second"] = decoder_input_ids2
    return instance



def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def get_emotions(dataset):
    for data in tqdm(dataset['valid']):
        utterances = data['utterances']
        for utterance in utterances:
            true_emotion = utterance["emotion"]

def pad_step_sent(sent,max_length,pad_id,type="pad"):
    sent=sent[:max_length]
    if type=="end":#pad inputids
        for k, t in enumerate(sent):
            if t == EOS_TOKEN:
                end_i = k
                break
            else:
                end_i=max_length
        for k in range(max_length):
            if k > end_i:
                sent[k] = pad_id
    if type=="pad":#pad emo input
        if len(sent)>max_length:
            sent=sent[:max_length]
        else:
            sent=sent+[pad_id]*(max_length-len(sent))
    return sent

def reply_steps(tokenizer,reply):
    if reply==[]:
        return reply
    else :
        reply=[tokenizer.encode(s) for s in reply]
        return reply
def select_dict( concept ):
    if len(concept) <= 3:
        num = 3
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
def calculate_metrics(args,  model, tokenizer,emo_labels_dict, dataset):
    # special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    nrc_dict = pickle.load(open("data/kgs/NRC_VAD.pkl", "rb"))
    all_emo_length=[]
    all_emo_f1_scores = []
    all_topic_length=[]
    all_topic_f1_scores = []
    all_final_length=[]
    all_final_f1_scores = []
    all_emo_sentences = []
    all_topic_sentences = []
    all_final_sentences = []
    all_pre_emo_cls=[]
    all_true_emo_cls=[]

    f= open(os.path.join(args.result_path,'test_results.txt'), 'a')
    # result_f=open(os.path.join(args.result_path,'alto_eval_results.txt'), 'a')
    for data in tqdm(dataset['test']):
        utterances = data['utterances']
        for utterance in utterances:
            concept = utterance["concept_net"]
            history = utterance["history"]
            history = " ".join(history)
            bart_concept=select_dict(concept)
            # concept_words = ["Ġ" + t[0] for k, v in bart_concept.items() for t in v]
            history_tokens, emo_intensity,knowledge_confidence,history_pos, vm, tag,token_type = add_knowledge_with_vm([history.strip()], bart_concept,nrc_dict,
                                                                         tokenizer, add_pad=True,
                                                                         max_length=MAX_LENGTH[0]+2)
            history_token_ids= [tokenizer.convert_tokens_to_ids(t) for t in history_tokens[0]]
            token_type_ids = [tokenizer.convert_tokens_to_ids(t) for t in token_type[0]]
            # emo_input_ids=[NRC_CLASS[t] for t in emo_input_tokens[0]]
            tag = [[1 if i == 0 else 0 for i in tag[0]]]
            emotion = emo_labels_dict[utterance["emotion"][0]]
            all_true_emo_cls.append(emotion)
            input_ids = torch.tensor(history_token_ids, device=args.device).unsqueeze(0)
            token_type_ids=torch.tensor(token_type_ids,device=args.device).unsqueeze(0)
            emo_intensity = torch.tensor(emo_intensity, device=args.device).unsqueeze(0)
            knowledge_confidence = torch.tensor(knowledge_confidence, device=args.device).unsqueeze(0)
            vm = torch.tensor(vm, device=args.device)#.unsqueeze(0)
            attention_mask=torch.tensor([0 if ids == PAD_TOKEN else 1 for ids in history_token_ids], device=args.device).unsqueeze(0)
            tag=torch.tensor(tag,device=args.device)
            position_ids = torch.tensor(history_pos, device=args.device)#.unsqueeze(0)
            predicted_emo_fdbk = []
            predicted_topic_fdbk=[]
            predicted_final_sent=[]
            decoder_hidden_states_emo=[]
            decoder_hidden_states_topic=[]
            for i in range(MAX_LENGTH[2]+1):
                instance= build_input_from_segments(nrc_dict,predicted_emo_fdbk,predicted_topic_fdbk, emotion,tokenizer, with_eos=False)
                decoder_input_ids_first=torch.tensor(instance["decoder_input_ids_first"], device=args.device).unsqueeze(0)
                decoder_input_ids_second = torch.tensor(instance["decoder_input_ids_second"], device=args.device).unsqueeze(0)
                if i<MAX_LENGTH[2]:
                    emo_fdbk_logits, topic_fdbk_logits,\
                    emo_logits,emo_decoder_states,topic_decoder_states= model(input_ids=input_ids,
                                                                   # emo_input_ids=None,#emo_input_ids,
                                                                              emo_intensity=emo_intensity,
                                                                              knowledge_confidence=knowledge_confidence,
                                                                              token_type_ids=token_type_ids,
                                                                position_ids=position_ids,
                                                                vm=vm,
                                                                attention_mask=tag,
                                                                decoder_input_ids_first=decoder_input_ids_first,
                                                                decoder_input_ids_second=decoder_input_ids_second,
                                                                decoder_input_ids_final=None,
                                                                decoder_attention_mask_first=None,
                                                                decoder_attention_mask_second=None,
                                                                decoder_attention_mask_final=None,
                                                                is_train=False,
                                                                is_integrate=False,
                                                                              hard_attention=False,
                                                                              )
                    if i<MAX_LENGTH[1]:
                        decoder_hidden_states_emo.append(emo_decoder_states[:,-1,:])
                    if i<MAX_LENGTH[2]:
                        decoder_hidden_states_topic.append(topic_decoder_states[:,-1,:])
                    emo_fdbk_logits = emo_fdbk_logits[0, -1, :] #/ args.temperature  # 取最新生成的词语
                    emo_fdbk_logits = top_filtering(emo_fdbk_logits, top_k=args.top_k, top_p=args.top_p)
                    emo_fdbk_probs = F.softmax(emo_fdbk_logits, dim=-1)
                    topic_fdbk_logits = topic_fdbk_logits[0, -1, :] #/ args.temperature  # 取最新生成的词语
                    topic_fdbk_logits = top_filtering(topic_fdbk_logits, top_k=args.top_k, top_p=args.top_p)
                    topic_fdbk_probs = F.softmax(topic_fdbk_logits, dim=-1)
                    emo_fdbk_prev = torch.topk(emo_fdbk_probs, 1)[1] if args.no_sample else torch.multinomial(
                        emo_fdbk_probs, 1)
                    topic_fdbk_prev = torch.topk(topic_fdbk_probs, 1)[1] if args.no_sample else torch.multinomial(
                        topic_fdbk_probs, 1)
                    predicted_emo_fdbk.append(emo_fdbk_prev.item())
                    predicted_topic_fdbk.append(topic_fdbk_prev.item())
                    # predicted_emo_fdbk_tokens.append(tokenizer.decode(emo_fdbk_prev.item()))
                else:
                    decoder_input_ids_first=pad_step_sent(predicted_emo_fdbk,MAX_LENGTH[1],PAD_TOKEN,type="end")
                    decoder_input_ids_second=pad_step_sent(predicted_topic_fdbk,MAX_LENGTH[2],PAD_TOKEN,type="end")
                    decoder_atten_mask_first = torch.tensor(
                        [0 if ids == PAD_TOKEN else 1 for ids in decoder_input_ids_first],
                        device=args.device).unsqueeze(0)
                    decoder_atten_mask_second = torch.tensor(
                        [0 if ids == PAD_TOKEN else 1 for ids in decoder_input_ids_second],
                        device=args.device).unsqueeze(0)
                    decoder_input_ids_first = torch.tensor(decoder_input_ids_first, device=args.device).unsqueeze(0)
                    decoder_input_ids_second = torch.tensor(decoder_input_ids_second, device=args.device).unsqueeze(0)
                    decoder_hidden_states_emo=torch.stack(decoder_hidden_states_emo).transpose(0,1)
                    decoder_hidden_states_topic=torch.stack(decoder_hidden_states_topic).transpose(0,1)
                    decoder_hidden_states=(decoder_hidden_states_emo,decoder_hidden_states_topic)
                    for j in range(MAX_LENGTH[3]):
                        decoder_input_ids_final = [BOS_TOKEN] + predicted_final_sent
                        decoder_input_ids_final = torch.tensor(decoder_input_ids_final, device=args.device).unsqueeze(0)
                        final_sent_logits = model(input_ids=input_ids,
                                                  token_type_ids=token_type_ids,
                                                                # emo_input_ids=None,#emo_input_ids,
                                                                position_ids=position_ids,
                                                                emo_intensity=emo_intensity,
                                                                knowledge_confidence=knowledge_confidence,
                                                                vm=vm,
                                                                attention_mask=tag,
                                                                decoder_input_ids_first=decoder_input_ids_first,
                                                                decoder_input_ids_second=decoder_input_ids_second,
                                                                decoder_input_ids_final=decoder_input_ids_final,
                                                                decoder_attention_mask_first=decoder_atten_mask_first,
                                                                decoder_attention_mask_second=decoder_atten_mask_second,
                                                                decoder_attention_mask_final=None,#decoder_atten_mask_final,
                                                                decoder_inputs_embeds=decoder_hidden_states,
                                                                is_train=False,
                                                                is_integrate=True,
                                                  hard_attention=False)  # 进入model的forward
                        final_sent_logits = final_sent_logits[0, -1, :]  # 取最新生成的词语
                        final_sent_logits = top_filtering(final_sent_logits, top_k=args.top_k, top_p=args.top_p)
                        final_sent_probs = F.softmax(final_sent_logits, dim=-1)
                        final_sent_prev = torch.topk(final_sent_probs, 1)[1] if args.no_sample else torch.multinomial(
                            final_sent_probs, 1)
                        if final_sent_prev.item()  in [PAD_TOKEN,EOS_TOKEN]:
                            break
                        predicted_final_sent.append(final_sent_prev.item())
            emo_predicted_sentence = tokenizer.decode(predicted_emo_fdbk, skip_special_tokens=True)
            emo_true_sentence =  " ".join(utterance["reply_first"]).strip()#tokenizer.decode(emo_fdbk_true_label, skip_special_tokens=True)
            topic_predicted_sentence = tokenizer.decode(predicted_topic_fdbk, skip_special_tokens=True)
            topic_true_sentence = " ".join(utterance["reply_second"]).strip()#tokenizer.decode(topic_fdbk_true_label, skip_special_tokens=True)
            final_predicted_sentence = tokenizer.decode(predicted_final_sent, skip_special_tokens=True)
            final_true_sentence =  " ".join(utterance["reply"]).strip()#tokenizer.decode(final_sent_true_label, skip_special_tokens=True)
            history_sentence=history.lstrip("<s>").rstrip("</s>").strip()#tokenizer.decode(history,skip_special_tokens=True)
            f.write(history_sentence + '\n')
            f.write(emo_true_sentence + '\n')
            f.write(emo_predicted_sentence + '\n')
            f.write(topic_true_sentence + '\n')
            f.write(topic_predicted_sentence + '\n')
            f.write(final_true_sentence+'\n')
            f.write(final_predicted_sentence + '\n')
            f.write('\n')
            emo_length=len(emo_predicted_sentence.strip().split(' '))
            emo_f1_score = _f1_score(emo_predicted_sentence, [emo_true_sentence])
            topic_length = len(topic_predicted_sentence.strip().split(" "))
            topic_f1_score = _f1_score(topic_predicted_sentence, [topic_true_sentence])
            final_length = len(final_predicted_sentence.strip().split(" "))
            final_f1_score = _f1_score(final_predicted_sentence, [final_true_sentence])
            all_emo_sentences.append(
                [nltk.word_tokenize(emo_predicted_sentence.strip()), nltk.word_tokenize(emo_true_sentence.strip())])
            all_topic_sentences.append(
                [nltk.word_tokenize(topic_predicted_sentence.strip()), nltk.word_tokenize(topic_true_sentence.strip())])
            all_final_sentences.append(
                [nltk.word_tokenize(final_predicted_sentence.strip()), nltk.word_tokenize(final_true_sentence.strip())])
            #print(f1_score)
            all_emo_length.append(emo_length)
            all_emo_f1_scores.append(emo_f1_score)
            all_topic_length.append(topic_length)
            all_topic_f1_scores.append(topic_f1_score)
            all_final_length.append(final_length)
            all_final_f1_scores.append(final_f1_score)
    all_step_sentences = []
    emo_distincts = eval.calc_distinct(all_emo_sentences)
    emo_bleus = eval.calc_bleu(all_emo_sentences)
    topic_distincts = eval.calc_distinct(all_topic_sentences)
    topic_bleus = eval.calc_bleu(all_topic_sentences)
    final_distincts = eval.calc_distinct(all_final_sentences)
    final_bleus = eval.calc_bleu(all_final_sentences)
    for i, s in enumerate(zip(all_emo_sentences, all_topic_sentences,all_final_sentences)):
        all_step_sentences.append([s[0][0]  + s[1][0],s[2][1]])
    step_distincts = eval.calc_distinct(all_step_sentences)
    step_bleus = eval.calc_bleu(all_step_sentences)
    print("avg stepsent bleu", step_bleus)
    print("avg stepsent distinct", step_distincts)
    #compare predicted and label with bleu
    print("avg emofdbk length", np.mean(all_emo_length))
    print("avg emofdbk f1 score", np.mean(all_emo_f1_scores))
    print("avg emofdbk bleu", emo_bleus)
    print("avg emofdbk distinct", emo_distincts)
    print("avg topicfdbk length", np.mean(all_topic_length))
    print("avg topicfdbk f1 score", np.mean(all_topic_f1_scores))
    print("avg topicfdbk bleu", topic_bleus)
    print("avg topicfdbk distinct", topic_distincts)
    print("avg finalsent length", np.mean(all_final_length))
    print("avg finalsent f1 score", np.mean(all_final_f1_scores))
    print("avg finalsent bleu", final_bleus)
    print("avg finalsent distinct", final_distincts)


def run():
    config_file = "configs/test_pipline_config.json"
    config = InteractConfig.from_json_file(config_file)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(config))

    if config.model_checkpoint == "":
        config.model_checkpoint = download_pretrained_model()

    random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    emo_labels_dict = {}
    for i, emo in enumerate(EMO_LABELS):
        emo_labels_dict[emo] = i
    logger.info("Get pretrained model and tokenizer")
    tokenizer = BartTokenizer.from_pretrained(config.model_checkpoint)
    # tokenizer.set_special_tokens(SPECIAL_TOKENS)
    model_class = StepBartForDialogueGeneration
    model = model_class.from_pretrained(config.model_checkpoint)
    # model.set_num_special_tokens(SPECIAL_TOKENS)
    model.to(config.device)

    model.eval()

    # dataset = get_dataset(tokenizer, config.dataset_path, config.dataset_cache)
    with open(config.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())

    calculate_metrics(config, model, tokenizer,emo_labels_dict,  dataset)

if __name__ == "__main__":
    run()
