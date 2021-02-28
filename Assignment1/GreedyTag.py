from HMMTag import *


def greedy_search(data_file_name):
    preds = []

    with open(data_file_name, 'r') as f:
        for line in f:
            sentence_tags = []
            line = line.replace("\n", "").split(" ")

            prev_tag = "START"
            prev_prev_tag = "START"

            for word_idx, word in enumerate(line):
                curr_word_tag_count = hmm_data["emissions"].get(word, hmm_data["emissions"]['*UNK*'])
                if word_idx == 0:
                    prev_prev_tag = 'START'
                    prev_tag = 'START'

                min_score = np.inf
                min_tag = ''
                for curr_tag in curr_word_tag_count.keys():
                    score = getQ(prev_prev_tag, prev_tag, curr_tag) + getE(word, curr_tag)
                    if score <= min_score:
                        min_score = score
                        min_tag = curr_tag
                sentence_tags.append(f'{word}/{min_tag}')
                prev_prev_tag = prev_tag
                prev_tag = min_tag
            preds.append(sentence_tags)

    return preds


if __name__ == "__main__":
    input_file = sys.argv[1]
    q_mle = sys.argv[2]
    e_mle = sys.argv[3]
    greedy_hmm_output = sys.argv[4]
    extra_file = sys.argv[5]
    load_estimates(q_mle, e_mle)
    preds_ = greedy_search(data_file_name=input_file)
    calc_accuracy(preds_, y_file_path='ass1-tagger-dev')
    save_predictions(preds_, greedy_hmm_output)
