import numpy as np


def load_and_normalize_vectors(file):
    W = []
    words = []
    with open(file, 'r', encoding='utf8') as f:
        for row in f:
            row = row.split()
            words.append(row[0])
            v = np.array(row[1:], dtype=np.float)
            W.append(v / np.linalg.norm(v))

    return np.array(W), np.array(words)


def get_top20_word_similarity():
    bow_vecs_file = 'word2vec/bow5.words'
    dep_vecs_file = 'word2vec/deps.words'

    bow_W, bow_words = load_and_normalize_vectors(bow_vecs_file)
    dep_W, dep_words = load_and_normalize_vectors(dep_vecs_file)

    print(f'bow5 words: {len(bow_words)}')
    print(f'bow5 matrix shape: {bow_W.shape}')
    print(f'dep words: {len(dep_words)}')
    print(f'dep matrix shape: {dep_W.shape}')

    # W and words are numpy arrays.
    bow_w2i = {w: i for i, w in enumerate(bow_words)}
    dep_w2i = {w: i for i, w in enumerate(dep_words)}

    target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar',
                    'piano']
    file = 'word2vec_top20.txt'
    with open(file, 'w', encoding='utf8') as f:
        for target in target_words:
            bow_v = bow_W[bow_w2i[target]]  # get target vector
            dep_v = dep_W[dep_w2i[target]]  # get target vector

            bow_sims = bow_W.dot(bow_v)
            dep_sims = dep_W.dot(dep_v)

            most_similar_ids = bow_sims.argsort()[-2:-22:-1] # start, end, step
            bow_sim_words = bow_words[most_similar_ids]

            most_similar_ids = dep_sims.argsort()[-2:-22:-1] # start, end, step
            dep_sim_words = dep_words[most_similar_ids]

            f.write(f"\\captionof{{table}}{{Top 20 similar words for: {target}}}\n")
            f.write('\\begin{tabular}{ c|c }\n')
            f.write(f'\\hline\n')
            f.write(f'BoW-5 & Dependency edge \\\\\n')
            f.write(f'\\hline\n')

            for w1, w2 in zip(bow_sim_words, dep_sim_words):
                f.write(f"{w1:<20}&{w2}\\\\\n")
            f.write(f'\\hline\n')
            f.write('\\end{tabular}\n')


def get_top10_attributes():
    bow_vecs_file = 'word2vec/bow5.words'
    dep_vecs_file = 'word2vec/deps.words'

    bow_W, bow_words = load_and_normalize_vectors(bow_vecs_file)
    dep_W, dep_words = load_and_normalize_vectors(dep_vecs_file)

    print(f'bow5 words: {len(bow_words)}')
    print(f'bow5 matrix shape: {bow_W.shape}')
    print(f'dep words: {len(dep_words)}')
    print(f'dep matrix shape: {dep_W.shape}')

    bow_att_vecs_file = 'word2vec/bow5.contexts'
    dep_att_vecs_file = 'word2vec/deps.contexts'

    bow_att_W, bow_att = load_and_normalize_vectors(bow_att_vecs_file)
    dep_att_W, dep_att = load_and_normalize_vectors(dep_att_vecs_file)

    print(f'bow5 att: {len(bow_att)}')
    print(f'bow5 matrix shape: {bow_att_W.shape}')
    print(f'dep att: {len(dep_att)}')
    print(f'dep matrix shape: {dep_att_W.shape}')

    # W and words are numpy arrays.
    bow_w2i = {w: i for i, w in enumerate(bow_words)}
    dep_w2i = {w: i for i, w in enumerate(dep_words)}

    # bow_att2i = {w: i for i, w in enumerate(bow_att)}
    # dep_att2i = {w: i for i, w in enumerate(dep_att)}
    target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar',
                    'piano']

    file = 'word2vec_top10_att.txt'
    with open(file, 'w', encoding='utf8') as f:
        for target in target_words:
            bow_v = bow_W[bow_w2i[target]]  # get target vector
            dep_v = dep_W[dep_w2i[target]]  # get target vector

            bow_att_sims = bow_att_W.dot(bow_v)
            dep_att_sims = dep_att_W.dot(dep_v)

            most_similar_ids = bow_att_sims.argsort()[-1:-11:-1]
            bow_sim_att = bow_att[most_similar_ids]

            most_similar_ids = dep_att_sims.argsort()[-1:-11:-1]
            dep_sim_att = dep_att[most_similar_ids]

            f.write(f"\\captionof{{table}}{{Top 10 attributes for: {target}}}\n")
            f.write('\\begin{tabular}{ c|c }\n')
            f.write(f'\\hline\n')
            f.write(f'BoW-5 & Dependency edge \\\\\n')
            f.write(f'\\hline\n')

            for w1, w2 in zip(bow_sim_att, dep_sim_att):
                f.write(f"{w1:<20}&{w2}\\\\\n")
            f.write(f'\\hline\n')
            f.write('\\end{tabular}\n')


get_top20_word_similarity()
get_top10_attributes()
