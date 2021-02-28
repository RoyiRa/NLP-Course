import itertools

import stanza
from spacy_stanza import StanzaLanguage

from features_extraction import dependency_path, words_between
from utils import read_annotations
import random
import pandas as pd

processor_dict = {
    'tokenize': 'default',
    'pos': 'default',
    'ner': 'conll03',
    'lemma': 'default'
}
snlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True, processors=processor_dict)
nlp = StanzaLanguage(snlp)

def get_dep_path(sent, per, org):
    for p in [ent for ent in sent.ents if ent.label_ == 'PER']:
        if p.text == per:
            per = p
            break
    for o in [ent for ent in sent.ents if ent.label_ == 'ORG']:
        if o.text == org:
            org = o
            break

    return dependency_path(per, org, root_pos=True)


def get_precision(gold_annotations, pred_annotations, entirety=False):
  per_miss = []
  TP, FP = 0, 0
  for sent_id, annotations in pred_annotations.items():
      for pred_ann in annotations:
          per, _, org, sent = pred_ann
          T = False
          for glod_ann in gold_annotations[sent_id]:
              if entirety:
                  if per == glod_ann[0] and org == glod_ann[2]:
                      T = True
                      break
              else:
                  if (per in glod_ann[0] or glod_ann[0] in per) and (org in glod_ann[2] or glod_ann[2] in org):
                      T = True
                      break
          if T:
              TP += 1
          else:
              FP += 1
              per_miss.append((sent_id, per, org, sent))
  return TP / (TP+FP), per_miss


def get_recall(gold_annotations, pred_annotations, entirety=False):
  recall_miss = []
  TP, FN = 0, 0
  for sent_id, annotations in gold_annotations.items():
      for gold_ann in annotations:
          per, _, org, sent = gold_ann
          T = False
          for pred_ann in pred_annotations[sent_id]:
              if entirety:
                  if per == pred_ann[0] and org == pred_ann[2]:
                      T = True
                      break
              else:
                  if (per in pred_ann[0] or pred_ann[0] in per) and (org in pred_ann[2] or pred_ann[2] in org):
                      T = True
                      break
          if T:
              TP += 1
          else:
              FN += 1
              recall_miss.append((sent_id, per, org, sent))
  return TP / (TP+FN), recall_miss


def eval(train_file):
    print('TRAIN')
    gold_annotations = read_annotations('../data/TRAIN.annotations', return_sent=True)
    pred_annotations = read_annotations(train_file, return_sent=True)

    print(f'Entities extracted as entirety')
    P, P_miss = get_precision(gold_annotations, pred_annotations, entirety=True)
    R, R_miss = get_recall(gold_annotations, pred_annotations, entirety=True)
    print(f'Precision: {P:.3f}')
    print(f'Recall:    {R:.3f}')
    print(f'F1:        {((2 * R * P) / (P + R)):.3f}')

    print()
    print(f'Entities extracted with "in" method')
    P, P_miss = get_precision(gold_annotations, pred_annotations, entirety=False)
    R, R_miss = get_recall(gold_annotations, pred_annotations, entirety=False)
    print(f'Precision: {P:.3f}')
    print(f'Recall:    {R:.3f}')
    print(f'F1:        {((2 * R * P) / (P + R)):.3f}')

    print(f'precision miss:')
    for t in random.sample(P_miss, 5):
        print(t)
        print(get_dep_path(sent=nlp(t[3][2:-2]), per=t[1], org=t[2]))
    print()
    print(f'recall miss:')
    for t in random.sample(R_miss, 5):
        print(t)
        print(get_dep_path(sent=nlp(t[3][2:-2]), per=t[1], org=t[2]))


# eval('../train_relations_w_rules.txt')
# print()


def rule_retired(annotations_file):
    with open(annotations_file, 'r', encoding="utf8") as in_f:
        for line in in_f:
            sent_id, person, _, org, sent = line.split('\t')

            tokens = sent.split()
            if 'retired' in tokens:
                p_idx = tokens.index(person.split()[-1])
                o_idx = tokens.index(org.split()[-1])
                r_idx = tokens.index('retired')
                if (p_idx < r_idx < o_idx) or (o_idx < r_idx < p_idx):
                    if r_idx + 2 >= o_idx:
                        print(sent)

# print('RETIRED RULE')
# rule_retired('../train_relations.txt')

def rule_org_s(annotations_file):
    with open('../data/lexicon.location', 'r', encoding='utf8') as f:
        lex_loc = set([loc.strip() for loc in f])
    with open(annotations_file, 'r', encoding="utf8") as in_f:
        for line in in_f:
            sent_id, person, _, org, sent = line.split('\t')

            tokens = sent.split()
            if "'s" in tokens:
                p_idx = tokens.index(person.split()[-1])
                o_idx = tokens.index(org.split()[-1])
                s_idx = tokens.index("'s")
                if o_idx < s_idx < p_idx:
                    if s_idx - 1 == o_idx:
                        if org in lex_loc:
                            print(sent)

# print('ORG S RULE')
# rule_org_s('../train_relations.txt')





def double_workplace(annotations_file):
    output_file_name = annotations_file.split('.')
    with open(annotations_file, 'r', encoding="utf8") as in_f:
        data = [line.strip().split('\t')[0:5] for line in in_f]
    with open(output_file_name[0] + 'new_rule.txt', 'w', encoding="utf8") as out_f:
        gby = pd.DataFrame(data, columns=['sentid', 'per', 'rel', 'org', 'sent']).groupby(by=['sentid', 'per'])
        for g in gby.groups:
            df = gby.get_group(g)
            if df.shape[0] > 1:
                sent_id, sent, per = df[['sentid', 'sent', 'per']].values[0]
                print(sent_id, sent)
                sent = nlp(sent)

                for p in [ent for ent in sent.ents if ent.label_ == 'PER']:
                    if p.text == per:
                        per = p
                        break
                org_ents = [o for org in df['org'] for o in [ent for ent in sent.ents if ent.label_ == 'ORG'] if o.text == org]

                print(per)
                print(org_ents)
                p_orgs = set()
                for org1, org2 in itertools.combinations(org_ents, r=2):
                    wb = words_between(org1, org2, sent)
                    if wb[0] == "'s":
                        p_orgs.update([org1, org2])

                for org in org_ents:
                    # if org.start-2 > 0 and sent[org.start-2].text == 'of':
                    #     p_orgs.add(org)
                    # if org.start-1 > 0 and sent[org.start-1].text == 'of':
                    #     p_orgs.add(org)
                    print(per, org)
                    dep_path, root_pos = dependency_path(per, org, root_pos=True)
                    print(dep_path)
                    print(root_pos)

                print(p_orgs)

                    # if 'nmod' in dep_path[-3:]:
                    #     out_f.write(f'{sent_id}\t{per}\tWork_For\t{org}\t{sent.text}\n')
                    #     print(True)
                    # print()
                print()
# double_workplace('../train_relations.txt')

sent = nlp('W. Dale Nelson covers the White House for The Associated Press .')
persons = [ent for ent in sent.ents if ent.label_ == 'PER']
orgs = [ent for ent in sent.ents if ent.label_ == 'ORG']
print(f'persons: {persons}')
print(f'orgs: {orgs}')
for p, o in itertools.product(persons, orgs):
    print(p, o)
    print(dependency_path(p, o, root_pos=True))
    print()