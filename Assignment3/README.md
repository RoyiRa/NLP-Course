# BIU-NLP-Course
An assignment from the Natural Language Processing course in BIU, taught by Prof. Yoav Goldberg, Prof. Ido Dagan and Prof. Reut Tsarfaty - if you can, I **HIGHLY** recommend signing up for the class.

For the [technical report](https://github.com/RoyiRa/NLP-Course/blob/main/Assignment3/Report.pdf) 


For our third assignment, we were tasked with exploring word similarities and writing a report:

1. Filter out function-words
2. To compute word-similarities, use: cosine-similarity, PMI (word-feature association-measure)
3. Experiment with three types of distributional vectors for representing a target word: 
   1. Frequency of each content-word co-occurring with the target word **in the same sentence**
   2. Frequency of each content-word co-occurring with the target word **within a window of two content-words (per side)**
   3. Content-words which are connected to the target word by a dependency-edge (in addition to the connected word, the feature will include the label and direction of the dependency bwtween the target-word and the feature)
4. For a closed-set of words and for each type of distributional-vector:
   1. Evaluate the top 20 most similar words
   2. Manually label if the word is topically related to the target word
   3. Manually label if the word is semantically related to the target word
   4. Compute the Mean Average Precision (MAP) results for the three co-occurrence types (AP for each of the two words and their average as MAP)
6. Do the same with a W2V model
7. Compare between the two approaches
