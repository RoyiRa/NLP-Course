# BIU-NLP-Course
An assignment from the Natural Language Processing course in BIU, taught by Prof. Yoav Goldberg, Prof. Ido Dagan and Prof. Reut Tsarfaty - if you can, I **HIGHLY** recommend signing up for the class.

For the [technical report](https://github.com/RoyiRa/NLP-Course/blob/main/Assignment2/Report.pdf) 


For our second assignment, we were tasked with writing language-rules and a report:

1. The rules need to generate a closed set of sentences (no under-generation)
2. No over-generation; they should not generate sentences which are not grammatical:
   1. Sentences that begin with "Is it true that..." need to be a coherent question, with a question mark.
   2. Distinguish between transitive and intransitive verbs (e.g., 'yawn' is intransitive, and you can't yawn at an object)
3. The sentences do not need to make sense
4. The rules need to be compatiable with words which were not included in the assignment, but belong to the same category
5. Attach weights to rules to increase the chance of generating a realistic sentence
6. We chose to expand our rules to cover the WH-word questions and Relative Clauses phenomena. 
   Sentences such as: 
   1. wh-wrd question: I wonder where Sally is.
   2. relative clause: The man kissed the lady that the president met
7. Of our own volition, we attempted to distinguish between nouns that can perform a verb (actors) and those who cannot (objects) 
