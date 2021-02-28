from collections import defaultdict

from utils import BOW5, DEPENDENCY

TOPIC = 'topic'
SEMANTIC = 'semantic'

relevant = defaultdict(lambda: defaultdict(lambda: defaultdict(bool)))
retrieved = defaultdict(lambda: defaultdict(list))

num = "W2V"
input_file = f'annotation_output'
with open(input_file, 'r', encoding='utf8') as f:
    for row in f:
        values = row.rstrip('\n').split()

        if len(values) == 1:
            t_word = values[0]
        elif values:
            for i in range(0, len(values), 3):
                w, t_a, s_a = values[i:i+3]
                relevant[t_word][TOPIC][w] = True if t_a == '+' else False
                relevant[t_word][SEMANTIC][w] = True if s_a == '+' else False

            retrieved[t_word][BOW5].append(values[0])
            retrieved[t_word][DEPENDENCY].append(values[3])


def prec_r(R, target_word, method, judge):
    SUM = 0
    for i in range(R):
        w = retrieved[target_word][method][i]
        SUM += 1 if relevant[target_word][judge][w] else 0
    return SUM / R


def AP(target_word, method, judge):
    K = len(retrieved[target_word][method])
    N = sum(relevant[target_word][judge].values())

    SUM = 0
    for R in range(1, K+1):
        w = retrieved[target_word][method][R-1]
        rel = 1 if relevant[target_word][judge][w] else 0
        SUM += prec_r(R=R, target_word=target_word, method=method, judge=judge) * rel

    return SUM / N


def MAP(method):
    mean_ap_topic = 0
    mean_ap_semantic = 0
    print(method)
    for word in relevant:
        ap_topic = AP(target_word=word, method=method, judge=TOPIC)
        mean_ap_topic += ap_topic
        ap_semantic = AP(target_word=word, method=method, judge=SEMANTIC)
        mean_ap_semantic += ap_semantic
        print(f'Word: {word}, results: {ap_topic}, {ap_semantic}')

    print(f'TOPIC - Mean AP: {mean_ap_topic / len(relevant)}')
    print(f'SEMANTIC - Mean AP: {mean_ap_semantic / len(relevant)}')
    print()

print(f'for {num}:')
MAP(method=BOW5)
MAP(method=DEPENDENCY)
