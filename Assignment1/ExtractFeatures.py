from MLETrain import *


def build_features(file_path, gazetteers=None):

    features = []
    ner_tags = set()
    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            line = line.split(" ")

            words_in_line = []
            tags_in_line = []

            for idx, unprocessed_word_tag in enumerate(line):
                pairs = unprocessed_word_tag.split("/")
                words_in_line.append("/".join(pairs[:-1]))
                tags_in_line.append(pairs[-1])
                ner_tags.add(pairs[-1])

            num_words = len(words_in_line)
            for idx in range(num_words):
                word_features = extract(words_in_line, idx, tags_in_line)
                curr_tag = tags_in_line[idx]
                word_features.insert(0, curr_tag)

                features.append(" ".join(word_features))

    return features


def extract(sent, i, prev_tags):
    def add_feature(name, value):
        features.append(f"{name}={value}")
    features = []
    add_feature("form", sent[i])
    total_len = len(sent)
    word_len = len(sent[i])

    if 3 <= word_len:
        suffix = sent[i][-3:]
        add_feature("suff", suffix)

        prefix = sent[i][:3]
        add_feature("pref", prefix)

    is_first = i == 0
    add_feature("is_first", is_first)

    is_last = i == total_len - 1
    add_feature("is_last", is_last)

    if 1 < i:
        pt = prev_tags[i - 1]
        add_feature("pt", pt)

        ppt = prev_tags[i - 2]
        add_feature("pptpt", ppt+pt)

        pw = sent[i - 1]
        add_feature("pw", pw)

        ppw = sent[i - 2]
        add_feature("ppw", ppw)

    elif i == 1:
        pt = prev_tags[i-1]
        add_feature("pt", pt)

        pw = sent[i - 1]
        add_feature("pw", pw)

    if i < total_len - 2:
        nw = sent[i + 1]
        add_feature("nw", nw)

        nnw = sent[i + 2]
        add_feature("nnw", nnw)
    elif i < total_len - 1:
        nw = sent[i + 1]
        add_feature("nw", nw)

    return features


def save_features(features, file_name="features_file"):
    with open(file_name, 'w') as f:
        for feature in features:
            f.write(f'{feature}\n')


def load_gazetteers(file_name):
    words = defaultdict(lambda: False)
    with open(file_name, "r") as f:
        for idx, line in enumerate(f):
            line = line.replace("\n", "")
            words[line.lower()] = True

    return words


if __name__ == "__main__":
    corpus_file = sys.argv[1]
    features_file = sys.argv[2]
    features_ = build_features(corpus_file)
    save_features(features_, features_file)
