from collections import defaultdict
import numpy as np
import sys
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

hmm_data = {}


def build_estimates(file_path):
    emissions = defaultdict(lambda: 0)
    transitions = defaultdict(lambda: 0)

    with open(file_path, 'r') as f:
        for line in f:
            old_tags = ["START", "START"]
            line = line.replace("\n", "")
            line = line.split(" ")
            line_len = len(line)

            for idx, word_tag in enumerate(line):
                pairs = word_tag.split("/")

                # transitions
                first_key = pairs[-1]
                transitions[first_key] += 1

                second_key = old_tags[0] + '/' + old_tags[1]
                transitions[second_key] += 1

                third_key = old_tags[0] + '/' + old_tags[1] + '/' + pairs[-1]
                transitions[third_key] += 1

                if idx == 0:
                    start_key = old_tags[0]
                    transitions[start_key] += 1

                old_tags[0] = old_tags[1]
                old_tags[1] = pairs[-1]

                if idx == line_len - 1:
                    end_key = "END"
                    transitions[end_key] += 1

                    second_end_key = old_tags[1] + '/' + 'END'
                    transitions[second_end_key] += 1

                    third_end_key = old_tags[0] + '/' + old_tags[1] + '/' + 'END'
                    transitions[third_end_key] += 1

                # emissions
                word_sig = get_word_signatures(["/".join(pairs[:-1])])
                for sig in word_sig:
                    sig_key = sig + '/' + pairs[-1]
                    emissions[sig_key] += 1

                emissions[word_tag] += 1

    unk_dict = defaultdict(lambda: 0)
    for k, v in emissions.items():
        if v == 1:
            tag = k.split("/")[-1]
            unk = '*UNK*' + '/' + tag
            unk_dict[unk] += 1

    emissions = {**emissions, **unk_dict}

    return emissions, transitions


def save_estimates(emissions, transitions, e_file_name='e.mle', q_file_name='q.mle'):
    with open(e_file_name, 'w') as f:
        for k, _ in emissions.items():
            pairs = k.split("/")
            f.write(f"{'/'.join(pairs[:-1])} {pairs[-1]}\t{emissions['/'.join(pairs[:-1]) + '/' + pairs[-1]]}\n")

    with open(q_file_name, 'w') as f:
        for k, _ in transitions.items():
            pairs = k.split("/")

            if len(pairs) == 3:
                f.write(f"{pairs[0]} {pairs[1]} {pairs[2]}\t{transitions['/'.join(pairs[:])]}\n")

            elif len(pairs) == 2:
                f.write(f"{pairs[0]} {pairs[1]}\t{transitions['/'.join(pairs[:])]}\n")

            elif len(pairs) == 1:
                f.write(f"{pairs[0]}\t{transitions[pairs[0]]}\n")

            else:
                print(f"Error! Pairs len is: {len(pairs)}")


def load_estimates(transitions_file_name, emissions_file_name):
    transitions = {}
    emissions = {}
    seen_unigrams = []
    sum_unigram_tags = 0

    with open(transitions_file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "").split("\t")
            value = int(line[1])

            if " " in line[0]:
                key = line[0].replace(" ", "/")
            else:
                key = line[0]
                if key not in seen_unigrams:
                    sum_unigram_tags += value
                    seen_unigrams.append(key)
                else:
                    print(f"Error! Key is already in unigram_set but is being processed again")

            transitions[key] = value

    with open(emissions_file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "").split("\t")
            word, tag = line[0].split(" ")
            if word not in emissions.keys():
                emissions[word] = {tag: int(line[1])}
            elif tag not in emissions[word].keys():
                emissions[word][tag] = int(line[1])
            else:
                print(f'Error: tag already in emission[word]: {tag} for word {word}')

    hmm_data["emissions"] = emissions
    hmm_data["transitions"] = transitions
    hmm_data["sum_unigram_tags"] = sum_unigram_tags
    hmm_data["all_tags"] = seen_unigrams


def save_predictions(predictions, file_name):
    with open(file_name, 'w') as f:
        for sentence in predictions:
            f.write(f'{" ".join(sentence)}\n')


def load_predictions(file_name):
    preds = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "").split(" ")
            preds.append(line)

    return preds


def get_word_signatures(word):

    plural_suffixes = ('s', 'es', 'ies', 'en', 'ice')
    assigned_signatures = []
    if type(word) == str:
        if "/" in word:
            res = word.split("/")
            res.append(word)
        else:
            res = [word]
    else:
        res = word

    for perm in res:
        if perm.isupper():
            uppercase_sig = '^Up'
            assigned_signatures.append(uppercase_sig)

        elif perm.islower():
            lowercase_sig = '^Lo'
            assigned_signatures.append(lowercase_sig)

        if 2 <= len(perm) and perm[0].isupper() and perm[1:].islower():
            upper_then_lowercase = '^Aa'
            assigned_signatures.append(upper_then_lowercase)

        if perm.isnumeric():
            if len(perm) == 4:
                four_digit_num = '^FourDigitNum'
                assigned_signatures.append(four_digit_num)
            elif len(perm) == 2:
                two_digit_num = '^TwoDigitNum'
                assigned_signatures.append(two_digit_num)
            else:
                num = '^Num'
                assigned_signatures.append(num)

        if 2 <= len(perm.lower()):
            if perm[-2:] == 'ed':
                ed_sig = '^Ed'
                assigned_signatures.append(ed_sig)

        elif 3 <= len(perm.lower()):
            if perm[-3:] == 'ing':
                ing_sig = '^Ing'
                assigned_signatures.append(ing_sig)

        if perm.endswith(plural_suffixes):
            plural_sig = '^Plural'
            assigned_signatures.append(plural_sig)

    return assigned_signatures


if __name__ == "__main__":
    input_file = sys.argv[1]
    q_mle = sys.argv[2]
    e_mle = sys.argv[3]
    emissions_, transitions_ = build_estimates(input_file)
    save_estimates(emissions_, transitions_, e_mle, q_mle)
