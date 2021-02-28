from MLETrain import *


def load_features(file_name):
    v = DictVectorizer(sparse=True)
    features = []
    labels = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("\n", "").split(" ")

            labels.append(line[0])
            line_as_dict = feature_list_to_dict(line, 1)
            features.append(line_as_dict)

    return v.fit_transform(features), v, labels


def feature_list_to_dict(entry, start_idx):
    entry_as_dict = {}
    entry_len = len(entry)
    for entry_idx in range(start_idx, entry_len):
        ftr, val = entry[entry_idx].split("=", maxsplit=1)
        entry_as_dict[ftr] = val
    return entry_as_dict


def save_feature_mapping(features, file_name):
    pickle.dump(features, open(file_name, "wb"))


def save_model(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))


if __name__ == "__main__":
    features_file = sys.argv[1]
    model_file = sys.argv[2]

    vectorized_features, features_map_, labels_ = load_features(features_file)
    save_feature_mapping(features_map_, "features_map_file") # POS
    model_ = LogisticRegression(multi_class='multinomial', max_iter=5000)
    model_.fit(vectorized_features, labels_)
    save_model(model_, model_file)

