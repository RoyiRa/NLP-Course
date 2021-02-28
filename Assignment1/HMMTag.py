from MLETrain import *


def load_predictions(file_name):
    preds = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "").split(" ")
            preds.append(line)

    return preds


def calc_accuracy(x, y_file_path):
    with open(y_file_path, 'r') as f:
        correct = 0
        total_words = 0
        for line_idx, line in enumerate(f):
            line = line.replace("\n", "").split(" ")
            for word_idx, word in enumerate(line):
                if word == x[line_idx][word_idx]:
                    if x[line_idx][word_idx].split("/")[-1] != 'O':
                        correct += 1
                        total_words += 1
                else:
                    total_words += 1
    print(f'correct is {correct}, total is: {total_words}')
    print(f"Total accuracy: {correct / total_words}")


def getE(wi, ti):
    word_tag_count = hmm_data["emissions"].get(wi, hmm_data["emissions"]['*UNK*'])
    word_freq = word_tag_count.get(ti, 0)

    e_score = 0
    if ti in hmm_data["transitions"].keys():
        e_score = word_freq / hmm_data["transitions"][ti]

    lambda_1 = 0.7
    lambda_2 = 0.3

    if word_freq <= 4:
        word_signatures = get_word_signatures(wi)
        sig_score = 0
        for sig in word_signatures:
            total_sig_count = sum(hmm_data["emissions"][sig].values())
            sig_tag_prob = (hmm_data["emissions"][sig].get(ti, 0) / total_sig_count)

            if 0 < sig_tag_prob:
                sig_score += -np.log(sig_tag_prob)

        e_score = -np.log(lambda_1 * e_score) + (lambda_2 * sig_score)
    else:
        e_score = -np.log(e_score)

    return e_score


def getQ(t1, t2, t3):

    # Viterbi
    lambda_1 = 0.05
    lambda_2 = 0.05
    lambda_3 = 0.9

    trigram = "/".join([t1, t2, t3])
    first_two_tags = "/".join([t1, t2])
    bigram = "/".join([t2, t3])
    unigram = t3

    trigram_prob = 0
    bigram_prob = 0

    if first_two_tags in hmm_data["transitions"].keys():
        trigram_prob = hmm_data["transitions"].get(trigram, 0) / hmm_data["transitions"].get(first_two_tags)
    if t2 in hmm_data["transitions"].keys():
        bigram_prob = hmm_data["transitions"].get(bigram, 0) / hmm_data["transitions"].get(t2)

    unigram_prob = hmm_data["transitions"].get(unigram, 0) / hmm_data.get("sum_unigram_tags")

    trigram_score = 0
    bigram_score = 0
    unigram_score = 0

    if trigram_prob != 0:
        trigram_score = trigram_prob * lambda_3
    if bigram_prob != 0:
        bigram_score = bigram_prob * lambda_2
    if unigram_prob != 0:
        unigram_score = unigram_prob * lambda_1

    return -np.log(trigram_score + bigram_score + unigram_score)


def viterbi(data_file_name):
    preds = []

    with open(data_file_name, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.replace("\n", "").split(" ")
            len_line = len(line)
            V = defaultdict(lambda: defaultdict(lambda: {}))
            B = defaultdict(lambda: defaultdict(lambda: {}))

            for u in hmm_data["all_tags"]:
                for v in hmm_data["all_tags"]:
                    if u == "START" and v == "START":
                        V[0][u][v] = 1
                    else:
                        V[0][u][v] = 0
                    B[0][u][v] = None

            for k in range(1, len_line):
                prev_word = line[k - 1]
                prev_word_t_count = hmm_data["emissions"].get(prev_word, hmm_data["emissions"]['*UNK*'])
                for u in prev_word_t_count.keys():
                    curr_word = line[k]
                    curr_word_t_count = hmm_data["emissions"].get(curr_word, hmm_data["emissions"]['*UNK*'])
                    for v in curr_word_t_count.keys():
                        min_score = np.inf
                        min_tag = None

                        if k == 1:
                            q_score = getQ("START", u, v)
                            e_score = getE(curr_word, v)

                            prev_score = V[k - 1]["START"][u]
                            score = q_score + e_score + prev_score
                            if score <= min_score:
                                min_score = score

                            V[k][u][v] = min_score
                            B[k][u][v] = "START"

                        else:
                            prev_prev_word = line[k - 2]

                            prev_prev_word_t_count = hmm_data["emissions"].get(prev_prev_word, hmm_data["emissions"]['*UNK*'])
                            for w in prev_prev_word_t_count.keys():
                                q_score = getQ(w, u, v)
                                e_score = getE(curr_word, v)
                                prev_score = V[k - 1][w][u]

                                score = q_score + e_score + prev_score
                                if score <= min_score:
                                    min_score = score
                                    min_tag = w

                            V[k][u][v] = min_score
                            B[k][u][v] = min_tag

            min_score = np.inf
            min_u_tag = None
            min_v_tag = None
            penultimate_word = line[len_line - 2]
            penultimate_t_word_count = hmm_data["emissions"].get(penultimate_word, hmm_data["emissions"]['*UNK*'])

            for u in penultimate_t_word_count.keys():
                last_word = line[len_line - 1]
                last_t_word_count = hmm_data["emissions"].get(last_word, hmm_data["emissions"]['*UNK*'])

                for v in last_t_word_count.keys():
                    last_score = V[len_line - 1][u][v]
                    q_score = getQ(u, v, "END")

                    score = q_score + last_score
                    if score <= min_score:
                        min_score = score
                        min_u_tag = u
                        min_v_tag = v

            tags = [min_v_tag, min_u_tag]
            for i, k in enumerate(range(len_line - 2, 0, -1)):
                tags.append(B[k+1][tags[i+1]][tags[i]])
            tags = tags[::-1]

            tagged_sentence = []
            for i in range(len_line):
                tagged_sentence.append(line[i] + '/' + tags[i])
            preds.append(tagged_sentence)

    return preds


if __name__ == "__main__":
    input_file = sys.argv[1]
    q_mle = sys.argv[2]
    e_mle = sys.argv[3]
    viterbi_hmm_output = sys.argv[4]
    extra_file = sys.argv[5]
    load_estimates(q_mle, e_mle)
    preds_ = viterbi(input_file)
    calc_accuracy(preds_, y_file_path='ass1-tagger-dev') # POS
    # calc_accuracy(preds_, y_file_path='dev') # NER
    save_predictions(preds_, viterbi_hmm_output)
