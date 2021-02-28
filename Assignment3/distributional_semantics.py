import time
from collections import defaultdict, Counter
import math
from utils import *


class WordsStats:
    def __init__(self, window, min_lemma_count):
        self.window = window
        self.corpus_word_count = Counter()
        self.word_counts = defaultdict(lambda: defaultdict(Counter))
        self.total = {SENTENCE: 0, WINDOW: 0, DEPENDENCY: 0}
        self.total_smooth = {SENTENCE: 0, WINDOW: 0, DEPENDENCY: 0}
        self.total_w = {SENTENCE: defaultdict(lambda: 0), WINDOW: defaultdict(lambda: 0),
                        DEPENDENCY: defaultdict(lambda: 0)}
        self.total_w_smooth = {SENTENCE: defaultdict(lambda: 0), WINDOW: defaultdict(lambda: 0),
                               DEPENDENCY: defaultdict(lambda: 0)}
        self.total_att = {SENTENCE: defaultdict(lambda: 0), WINDOW: defaultdict(lambda: 0),
                          DEPENDENCY: defaultdict(lambda: 0)}
        self.int2str = {}
        self.min_lemma_count = min_lemma_count
        self.min_att_count = MIN_ATT_COUNT
        self.num_most_common_att = NUM_MOST_COMMON_ATT
        self.dep_top_att = Counter()

    def _get_hash(self, s):
        hashed_s = hash(s)
        if hashed_s in self.int2str:
            return hashed_s
        self.int2str[hashed_s] = s

        return hashed_s

    def _get_s(self, hashed_s):
        return self.int2str[hashed_s]

    def fit(self, file):
        def tokenize(tagged_sent):
            return {h: v for h, v in zip(FIELDS_H, tagged_sent)}

        def read_sentences():
            with open(file, 'r', encoding='utf8') as f:
                sentence = []
                for raw_sent in f:
                    raw_sent = raw_sent.rstrip('\n')
                    if raw_sent:
                        sentence.append(tokenize(raw_sent.split('\t')))
                    else:
                        yield sentence
                        sentence = []

                yield sentence

        word_frequency = Counter()
        for sent in read_sentences():
            for row in sent:
                if WordsStats.is_content_word(row[POSTAG], row[LEMMA]):
                    word_frequency[row[LEMMA]] += 1

        for w, c in word_frequency.items():
            if c >= self.min_att_count:
                self.corpus_word_count[w] = c

        for sent in read_sentences():
            content_words, prep_words = WordsStats.get_content_and_prep_words(sent)
            self.words_co_occurring(content_words=content_words)
            self.words_dependency(tokenized_sentence=sent, content_words=content_words, prep_words=prep_words)

        self.filter_attributes()

        for method in self.word_counts:
            for hashed_w, w_c in self.word_counts[method].items():
                # cache p(u,*)
                self.total_w[method][hashed_w] = sum(w_c.values())

                self.total_w_smooth[method][hashed_w] = sum([v ** 0.75 for v in w_c.values()])

                # cache p(*,*)
                self.total[method] += self.total_w[method][hashed_w]
                self.total_smooth[method] += self.total_w_smooth[method][hashed_w]

                # cache p(*,att)
                for hashed_att, att_c in w_c.items():
                    self.total_att[method][hashed_att] += att_c

        return self

    @staticmethod
    def is_content_word(postag, lemma):
        postag_valid = postag in CONTENT_WORD_TAGS
        lemma_valid = lemma not in STOP_WORDS and lemma not in PUNCTUATION_LIST
        return postag_valid and lemma_valid

    @staticmethod
    def get_content_and_prep_words(tokenized_sentence):
        content_words, prep_words = [], []
        for row in tokenized_sentence:
            if WordsStats.is_content_word(row[POSTAG], row[LEMMA]):
                content_words.append(row)
            elif row[POSTAG] in PREPOSITION:
                prep_words.append(row)
        return content_words, prep_words

    def set_attribute(self, w, att, method):
        hashed_w = self._get_hash(s=w)
        hashed_att = self._get_hash(s=att)

        self.word_counts[method][hashed_w][hashed_att] += 1
        if method == DEPENDENCY:
            self.dep_top_att[hashed_att] += 1

    def words_dependency(self, tokenized_sentence, content_words, prep_words):
        def build_dependency_attribute(w):
            label, co_word = None, None
            parent_w = tokenized_sentence[int(w[HEAD]) - 1] if int(w[HEAD]) > 0 else None

            if parent_w and WordsStats.is_content_word(parent_w[POSTAG], parent_w[LEMMA]):
                if self.corpus_word_count[parent_w[LEMMA]] >= self.min_att_count:
                    label = w[DEPREL]
                    co_word = parent_w[LEMMA]

            elif parent_w in prep_words:
                parent_parent_w = tokenized_sentence[int(parent_w[HEAD]) - 1] if int(parent_w[HEAD]) > 0 else None
                if parent_parent_w and WordsStats.is_content_word(parent_parent_w[POSTAG], parent_parent_w[LEMMA]):
                    if self.corpus_word_count[parent_parent_w[LEMMA]] >= self.min_att_count:
                        label = f'{parent_w[DEPREL]}:{parent_w[LEMMA]}'
                        co_word = parent_parent_w[LEMMA]

            if label and co_word:
                att_w = (co_word, label, IN)
                self.set_attribute(w=w[LEMMA], att=att_w, method=DEPENDENCY)

                if self.corpus_word_count[co_word] >= self.min_lemma_count:
                    att_co_w = (w[LEMMA], label, OUT)
                    self.set_attribute(w=co_word, att=att_co_w, method=DEPENDENCY)

        for w in content_words:
            if self.corpus_word_count[w[LEMMA]] >= self.min_lemma_count:
                build_dependency_attribute(w)

    def filter_attributes(self):
        for method in self.word_counts:
            for hashed_w in self.word_counts[method]:
                c = self.word_counts[method][hashed_w]
                self.word_counts[method][hashed_w] = Counter({att: count for att, count
                                                              in c.most_common(n=self.num_most_common_att)})

    def words_co_occurring(self, content_words):
        for i, w in enumerate(content_words):
            if self.corpus_word_count[w[LEMMA]] >= self.min_lemma_count:

                low = i - self.window if i >= self.window else 0
                high = i + self.window+1 if i + self.window+1 <= len(content_words) else len(content_words)
                window = content_words[low:i] + content_words[i+1:high]

                for co_word in window:
                    if self.corpus_word_count[co_word[LEMMA]] >= self.min_att_count:
                        self.set_attribute(w=w[LEMMA], att=co_word[LEMMA], method=WINDOW)

                sentence = content_words[0:i] + content_words[i+1:]

                for co_word in sentence:
                    if self.corpus_word_count[co_word[LEMMA]] >= self.min_att_count:
                        self.set_attribute(w=w[LEMMA], att=co_word[LEMMA], method=SENTENCE)


class WordSimilarities:
    def __init__(self, stats: WordsStats, smooth_ppmi=True):
        self.smooth_ppmi = smooth_ppmi
        self.stats = stats
        self.l2_norm = defaultdict(lambda: defaultdict(lambda: 0.))
        self.word_vecs = defaultdict(lambda: defaultdict(lambda: Counter()))  # for efficient cosine similarity
        self.att_vecs = defaultdict(lambda: defaultdict(lambda: Counter()))  # for efficient cosine similarity

    def fit(self):
        for method in self.stats.word_counts:
            p_all, p_u, p_att = 0, 0, Counter()
            for w, counter in self.stats.word_counts[method].items():
                for att in counter:
                    if att not in p_att:
                        p_att[att] += self.p(att=att, method=method)
                    p_all += self.p(u=w, att=att, method=method)
                p_u += self.p(u=w, method=method)
            print(f'{method:<15} p1: {p_all:<20} p1: {p_u:<20} p1: {sum(p_att.values()):<20}')

        # pre process - create ppmi vector to each word and also calc the norm.
        for method in self.stats.word_counts:
            for w in self.stats.word_counts[method]:
                for att in self.stats.word_counts[method][w]:
                    ppmi = self.get_PPMI(u=w, att=att, method=method)
                    if ppmi:
                        self.l2_norm[method][w] += ppmi ** 2
                        self.word_vecs[method][w][att] = ppmi
                        self.att_vecs[method][att][w] = ppmi

                self.l2_norm[method][w] = math.sqrt(self.l2_norm[method][w])
        return self

    def get_PPMI(self, u, att, method):
        ppmi = math.log(
            self.p(u=u, att=att, method=method) / (self.p(u=u, method=method) * self.p(att=att, method=method))
        )

        return max(ppmi, 0)

    def p(self, method, u=None, att=None):
        if u is None and att is None:
            return 1
        if u is None:
            return self.stats.total_att[method][att] / self.stats.total[method]
        if att is None:
            if self.smooth_ppmi:
                return self.stats.total_w_smooth[method][u] / self.stats.total_smooth[method]
            else:
                return self.stats.total_w[method][u] / self.stats.total[method]

        return self.stats.word_counts[method][u][att] / self.stats.total[method]

    def get_cosine_similarities(self, target, method):
        hashed_target = hash(target)
        hashed_similarity_result = Counter()
        similarity_result = Counter()

        for att, u_att_ppmi in self.word_vecs[method][hashed_target].items():
            for v, v_att_ppmi in self.att_vecs[method][att].items():
                hashed_similarity_result[v] += (u_att_ppmi * v_att_ppmi)

        del hashed_similarity_result[hashed_target]

        for v, sim in hashed_similarity_result.items():
            similarity_result[self.stats.int2str[v]] = sim / (
                        self.l2_norm[method][hashed_target] * self.l2_norm[method][v])

        return similarity_result


if __name__ == '__main__':
    min_lemma_count_ = 100

    start_time_total = time.time()

    start_time = time.time()
    file_ = 'wikipedia.sample.trees.lemmatized'

    stats = WordsStats(window=2, min_lemma_count=min_lemma_count_).fit(file=file_)

    print(f'Finished fit stats {(time.time() - start_time):.3f} sec')

    start_time = time.time()
    word_sim = WordSimilarities(stats=stats, smooth_ppmi=True).fit()
    print(f'Finished fit Similarities {(time.time() - start_time):.3f} sec')

    file_ = f'{min_lemma_count_}_features/top20.txt'
    with open(file_, 'w', encoding='utf8') as f:
        for word in target_words:
            sent_sim = word_sim.get_cosine_similarities(target=word, method=SENTENCE).most_common(20)
            win_sim = word_sim.get_cosine_similarities(target=word, method=WINDOW).most_common(20)
            dep_sim = word_sim.get_cosine_similarities(target=word, method=DEPENDENCY).most_common(20)
            print(f'{word}\n')
            f.write(f'{word}\n\n')

            for (sent_w, sent_s), (win_w, win_s), (dep_w, dep_s) in zip(sent_sim, win_sim, dep_sim):
                print(f"{win_w:<20} {win_s:.3f}\t{sent_w:<20} {sent_s:.3f}\t{dep_w:<20} {dep_s:.3f}")

                f.write(f"{win_w:<20} {sent_w:<20} {dep_w:<20}\n")

            print(f'*********')
            f.write(f'*********\n')
    print(f'file {file_} written')

    file_ = f'{min_lemma_count_}_features/counts_words.txt'
    with open(file_, 'w', encoding='utf8') as f:
        f.writelines([f"{w} {count}\n" for w, count in stats.corpus_word_count.most_common(n=50)])
    print(f'file {file_} written')

    file_ = f'{min_lemma_count_}_features/counts_contexts_dep.txt'
    with open(file_, 'w', encoding='utf8') as f:
        f.writelines([f"{stats.int2str[att]} {count}\n" for att, count in stats.dep_top_att.most_common(n=50)])
    print(f'file {file_} written')

    file_ = f'{min_lemma_count_}_features/top20_latex.txt'
    with open(file_, 'w', encoding='utf8') as f:
        for word in target_words:
            sent_sim = word_sim.get_cosine_similarities(target=word, method=SENTENCE).most_common(20)
            win_sim = word_sim.get_cosine_similarities(target=word, method=WINDOW).most_common(20)
            dep_sim = word_sim.get_cosine_similarities(target=word, method=DEPENDENCY).most_common(20)

            f.write(f'\\captionof{{table}}{{Words similarity to {word}}}\n')
            f.write('\\begin{tabular}{ c|c|c }\n')
            f.write(f'\\hline\n')
            f.write(f'2-word window & Sentence window & Dependency edge \\\\\n')
            f.write(f'\\hline\n')

            for (sent_w, sent_s), (win_w, win_s), (dep_w, dep_s) in zip(sent_sim, win_sim, dep_sim):
                f.write(f"{win_w:<20}&{sent_w:<20}&{dep_w}\\\\\n")
            f.write(f'\\hline\n')
            f.write('\\end{tabular}\n')

        file_ = f'{min_lemma_count_}_features/top20_att_latex.txt'
        with open(file_, 'w', encoding='utf8') as f:
            for word in target_words:
                hashed_w = hash(word)
                sent_w_atts = word_sim.word_vecs[SENTENCE][hashed_w].most_common(20)
                win_w_atts = word_sim.word_vecs[WINDOW][hashed_w].most_common(20)
                dep_w_atts = word_sim.word_vecs[DEPENDENCY][hashed_w].most_common(20)

                f.write(f"\\captionof{{table}}{{Highest attribute's PPMI of {word}}}\n")
                f.write('\\begin{tabular}{ c|c|c }\n')
                f.write(f'\\hline\n')
                f.write(f'2-word window & Sentence window & Dependency edge \\\\\n')
                f.write(f'\\hline\n')

                for (sent_att, _), (win_att, _), (dep_att, _) in zip(sent_w_atts, win_w_atts, dep_w_atts):
                    sent_att = stats.int2str[sent_att]
                    win_att = stats.int2str[win_att]
                    dep_att = stats.int2str[dep_att]

                    f.write(f"{win_att:<20}&{sent_att:<20}&{', '.join(dep_att)}\\\\\n")
                f.write(f'\\hline\n')
                f.write('\\end{tabular}\n')

    for method_ in word_sim.word_vecs:
        print(f'{method_:<15} '
              f'words for similarity: {len(word_sim.word_vecs[method_]):<20} '
              f'atts for similarity: {len(word_sim.att_vecs[method_]):<20}')

    print(f'Finished time: {(time.time() - start_time_total):.3f} sec')
    print(f'Min lemma count was: {min_lemma_count_}')