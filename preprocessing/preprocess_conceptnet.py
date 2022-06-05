import json
import pickle
import csv
from collections import defaultdict
def to_pickle(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

edchat=json.load(open("ED/ED_candidates.json",encoding="utf-8"))
#处理conceptnet的常识数据集
from ast import literal_eval
def get_ngrams(utter, n):
    sep_utter=[]
    for s in utter:
        sep_s=s.split(" ")
        sep_utter.append(sep_s)
    utter=utter.split(" ")
    total = []
    for i in range(len(utter)):
        for j in range(i, max(i - n, -1), -1):
            total.append("_".join(utter[j:i + 1]))
    return total

def get_all_ngrams(examples, n):
    all_ngrams = []
    for ex in examples:
        for utterance in ex["utterances"]:
            utter=utterance["history"][0]#+utterance["reply"][0]
            all_ngrams.extend(get_ngrams(utter, n))
    return set(all_ngrams)
train=edchat["train"]
valid=edchat["valid"]
test=edchat["test"]
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default="concept")
parser.add_argument('--n', default=1)
args = parser.parse_args()
#
dataset = args.dataset
n = args.n
ngrams = get_all_ngrams(train+valid+test,n)
print("Loading conceptnet...")
csv_reader = csv.reader(open("./KB/assertions.csv", "r"), delimiter="\t")
concept_dict = defaultdict(set)

for i, row in enumerate(csv_reader):
    if i % 1000000 == 0:
        print("Processed {0} rows".format(i))

    lang = row[2].split("/")[2]
    if lang == 'en':
        c1 = row[2].split("/")[3]
        c1_lang=row[2].split("/")[2]
        c2 = row[3].split("/")[3]
        c2_lang=row[3].split("/")[2]
        r=row[1].split("/")[2]
        weight = literal_eval(row[-1])["weight"]
        if c1 in ngrams and c1_lang=='en':
            concept_dict[c1].add((c2,r, weight))
        if c2 in ngrams and c2_lang=='en':
            concept_dict[c2].add((c1,r, weight))
print("Saving concepts...")
to_pickle(concept_dict, "./data/kgs/{0}.pkl".format(dataset))
# edchat=json.load(open("ED/ED_step.json",encoding="utf-8"))
# concept_target= {"concept_words":[]}
# num_have=0
# num_total=0
# concept=load_pickle("./data/KB/{0}.pkl".format(dataset))
# for dataset_name, dataset in edchat.items():
#     for dialog in dataset:
#         for k , utterance in enumerate(dialog["utterances"]):
#             if k>0:
#                 num_total +=1
#                 if concept_word != []:
#                     num_have+=1
#                 concept_target["concept_words"].append(concept_word)
#             source=utterance["history"][0].split(" ")
#             concept_word=[]
#             for i,s in enumerate(source):
#                 if s.lower() in concept.keys():
#                     concept_word.append((i, s.lower()))
#
#                 else:
#                     continue