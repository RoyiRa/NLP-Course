from collections import defaultdict

input_file = 'annotation_input'
target_words = defaultdict(set)
with open(input_file, 'r', encoding='utf8') as f:
    for row in f:
        values = row.rstrip('\n').split()
        if len(values) == 1:
            t_word = values[0]
        else:
            target_words[t_word].update(values)

topic_annotation = defaultdict(lambda: defaultdict(bool))
semantic_annotation = defaultdict(lambda: defaultdict(bool))
for t_word, words in target_words.items():
    print(f'Please respond y/n defualt is y')
    for w in words:
        topic_res = input(f'Is {w} topic related to {t_word} ?(y): ')
        semantic_res = input(f'Is {w} semantic related to {t_word} ?(y): ')
        print()
        if topic_res == 'y' or topic_res == '':
            topic_annotation[t_word][w] = True
        else:
            topic_annotation[t_word][w] = False

        if semantic_res == 'y' or semantic_res == '':
            semantic_annotation[t_word][w] = True
        else:
            semantic_annotation[t_word][w] = False

input_file = 'annotation_input'
output_file = 'annotation_output'
target_words = defaultdict(set)
with open(input_file, 'r', encoding='utf8') as f, open(output_file, 'w', encoding='utf8') as f2:
    for row in f:
        values = row.rstrip('\n').split()
        if len(values) == 1:
            t_word = values[0]
            f2.write(row)
        else:
            s = ""
            for w in values:
                s += f'{w:<20}{"+" if topic_annotation[t_word][w] else "-"} {"+" if semantic_annotation[t_word][w] else "-":<5}'
            f2.write(f'{s}\n')
