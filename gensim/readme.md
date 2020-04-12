Jagrons:
1.Tokenization: Breaks up the document into words(usually space is a delimiter).

The core concepts of gensim are:
1. Document : A document is a str of Python 3.
2. Corpus : A collection of documents, used in two applications. First, input for training the model and Second, documents to be classified after the model is trained.
3. Vector : mathematical representation of a document.
4. Model : algorithm to transform one vector ffrom one representation to another.

Vector in detail:
To infer the latent structure in our corpus we need a way to represent the document in a way such that we can manipulate them mathematicaly.
Hurray solution if we infer each document as a vector of vectors.
Example: we have 3 features [f1,f2,f3] and 2 documents [d1, d2].
for d1: (1,0) -> f1 | (2,2.0) -> f2 | (3,5.0) -> f3 : Then this is termed as dense vector.
Commonly referred as (0,2.0,5.0) aka vector with 3 dimensions.
