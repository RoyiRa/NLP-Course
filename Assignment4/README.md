# Relation Extration 

# BIU-NLP-Course
An assignment from the Natural Language Processing course in BIU, taught by Prof. Yoav Goldberg, Prof. Ido Dagan and Prof. Reut Tsarfaty - if you can, I **HIGHLY** recommend signing up for the class.

For the [technical report](https://github.com/RoyiRa/NLP-Course/blob/main/Assignment4/Report.pdf) 

For our fourth assignment, we were tasked with implementing a Relationship Extraction model and writing a report:

We focused on the 'Work_For' relationship. 
For instance, the model receives as input: "Alice won Facebook's employee-of-the-month for the Relationship Extraction model she developed!" and outputs: (Work_For, Alice, Facebook); because Alice works at Facebook. 

To successfully perform this task, we first used spaCy to extract text insights such as Named Entities, PoS, etc... Then, we created relevant representational vectors (we ignored sentences without PERSON and ORGANIZATION entities, since a Work_For relationship can only be between them), and trained a classifier to detect whether a sentence contains the relationship or not. Using Recall and Precision as our performance metrics, we sampled errors our model made, and fine-tuned it with rules to beat our previous results.


### Software dependencies:
```bash
git clone https://github.com/shon-otmazgin/nlp_relation_extraction.git
pip install -r requirements.txt
```

### Train:
To train our Relation Extraction model, please run ```train.py``` with 3 files:
1. ```corpus``` file in format of ```sentid<TAB>sent```
2. ```annotaion``` file in format of ```sentid<TAB>ent1<TAB>rel<TAB>ent2```
3. ```vocab``` vectors file or SpaCy Vocab with vectors. We download ```glove.6B.300d.zip``` you can download it from [Here](http://nlp.stanford.edu/data/glove.6B.zip) (right click), and extract it to ```data/glove.6B.300d.txt```.

Note: It may take a while, depending on your resources(cpu/gpu) and the size of the vectors file.

Example:
```python train.py data/Corpus.TRAIN.txt data/TRAIN.annotations data/glove.6B.300d.txt```

Program's output is 1 pickle file named ```trained_model```

### Extract Relations (Inference)
To get relations from trained model, please run ```extract.py``` with 2 files:
1. ```corpus``` file in format of ```sentid<TAB>sent```
2. ```output_file``` your desired output file name where the extracted relation will be written

Important Note: ```extract.py``` assume ```trained_model``` file exist in the content folder. Either train a model or download pre trained model from [Here](https://drive.google.com/file/d/1MLE49Doxl7mvZop9uO4KCYuuwev12fB6/view?usp=sharing)

Example:
```python extract.py data/Corpus.DEV.txt dev_relations.txt```

### Evaluation
To evaluate the results with gold annotations file please run ```eval.py``` with 2 files:
1. gold annotations in the format of ```sentid<TAB>ent1<TAB>rel<TAB>ent2``` 
2. predicted relations in the format of ```sentid<TAB>ent1<TAB>rel<TAB>ent2```

Example:
```python eval.py data/DEV.annotations dev_relations.txt```

Program's output is the precision, recall and f1 scores.

