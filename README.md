The core concepts of gensim are:
1. Document : A document is a str of Python 3.

2. Corpus : A collection of documents, used in two applications.
            First - input for training the model, and 
            Second - documents to be classified after the model is trained.
            
3. Vector : mathematical representation of a document.
            Details - To infer the latent structure in our corpus we need a way to represent the document in a way such that we can manipulate them mathematicaly.
            Solution - if we infer each document as a vector of features.
            Example: we have 3 features [f1,f2,f3] and 2 documents [d1, d2].
            for d1: (1,0) -> f1 | (2,2.0) -> f2 | (3,5.0) -> f3, then this is termed as Dense vector.
            Commonly referred as [0,2.0,5.0] aka vector with 3 dimensions.

            for d2 [0.1, 1.9, 4.9] is ~ to d1, however this conclusion is entirely dependent on how correctly we choose our features.


4. Model : algorithm to transform one vector from one representation to another. It is an abstract term used for transformation of document representation from one form to another.
        Example: tf-idf model transforms vectord from BOW representation to a vector space freq counts are weighted according to rarity of each token in the corpus.
        Ref https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#sphx-glr-auto-examples-core-run-topics-and-transformations-py
         

FYI: 
1. tokenization: Breaks up the document into words(usually space is a delimiter).

2. BOW approach just records the count of tokens in a document and completely ignores the ordering.
excample corpus -> ['coffee', 'milk', 'sugar', 'spoon'] and d1 -> 'coffee milk milk' then vector -> [1,2,0,0]

Important: Two different document can end up having same vector representations.
            If any token appears that was not present in the training corpus, its discarded in vectorization, and they are given implicit zero value as a space saving measure.

Crucial: Right selection of Features (garbage in, garbage out...). In our context the mapping between feature and id is a dictionary.

Further pre-requisites: https://radimrehurek.com/gensim/auto_examples/index.html
1. Corpora & Vector Spaces
2. Topic Modeling & Transformation functions
3. Similarity Queries.             
 ######################################################################################
 Word2vec : Shallow neural network that uses large unannotated texts as inputs and learns the token semantic relationship in an unsupervised learning and outputs are one vector per word. These vectors remarkably have linear relationship that allows us to perform similarity functions.
 It is an improvement over Bag of Words which earlier failed to capture the semantic relationship as it focused on frequency of token occurances in a corpus.
 
 Memory:
 w2v model parameters are stored as numpy matrices. Each array is [ #vocabulary (controlled by min_count parameter) * #size(dimension parameter) ] ~ 4Kbs. Three such matrices are stored presently.
 Example: 100,000 words & 200 dimensions = 100000*200*4*3 = ~229Mb
 
 Limitation:
 1. Unable to infer vectors for unfamiliar words.(overcome this by FastText Model)
 
  ######################################################################################
  Doc2Vec: In gensim paragraph vector model is represented as Doc2Vec.
  There are 2 implementations:
  1. PV-DM : Distributed Memory Paragraph Vector (analogous to wv CBoW)
  2. PV-DBOW  : Distributed Bag of Words PV (analogous to wv SG)
  