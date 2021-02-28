from MLETrain import *
from ExtractFeatures import extract, load_gazetteers
from TrainModel import feature_list_to_dict
from HMMTag import calc_accuracy


def load_feature_mapping(feature_map_file_name):
    return pickle.load(open(feature_map_file_name, 'rb'))


def load_model(model_file_name):
    return pickle.load(open(model_file_name, 'rb'))


def get_dataset(file_name):
    sentences = []
    with open(file_name, 'r') as f:
        for line in f:
            words_per_line = []
            line = line.replace("\n", "").split(" ")
            len_line = len(line)

            for line_idx in range(len_line):
                words_per_line.append(line[line_idx])

            sentences.append(words_per_line)

    return sentences


def predict(sentences, model, feature_map, gazetteers=None):
    sentences_len = len(sentences)
    tags_per_sentence = [[] for _ in range(sentences_len)]

    len_longest_sentence = max([len(sentence) for sentence in sentences])
    for sentence_idx in range(len_longest_sentence):  # current sentence
        words_to_pred = []
        words_loc = []
        is_sentence_too_short = True
        for j in range(sentences_len):
            curr_sentence_len = len(sentences[j])
            if sentence_idx < curr_sentence_len:
                words_to_pred.append(sentences[j][sentence_idx])
                words_loc.append(j)
                is_sentence_too_short = False

        if is_sentence_too_short is False:
            features_dicts = []
            for k, word in enumerate(words_to_pred):
                word_loc_in_sentence = words_loc[k]
                word_as_features = extract(sentences[word_loc_in_sentence], sentence_idx,
                                           tags_per_sentence[word_loc_in_sentence])
                feature_dict = feature_list_to_dict(word_as_features, 0)
                features_dicts.append(feature_dict)

            vectors = feature_map.transform(features_dicts)
            preds = model.predict(vectors)

            for k, pred in enumerate(preds):
                tags_per_sentence[words_loc[k]].append(pred)

    tagged_sentences = []
    for i, sentence in enumerate(sentences):
        tagged_sentence = []
        for j, word in enumerate(sentence):
            tagged_sentence.append("/".join([word, tags_per_sentence[i][j]]))
        tagged_sentences.append(tagged_sentence)

    return tagged_sentences


if __name__ == "__main__":
    input_file_name = sys.argv[1]
    model_file_name = sys.argv[2]
    feature_map_file = sys.argv[3]
    output_file = sys.argv[4]

    feature_map_ = load_feature_mapping(feature_map_file)
    model_ = load_model(model_file_name)
    sentences_ = get_dataset(input_file_name)
    preds_ = predict(sentences_, model_, feature_map_)
    calc_accuracy(preds_, y_file_path='ass1-tagger-dev')
    save_predictions(preds_, output_file)
