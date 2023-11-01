# Installing the required package
!pip install sentence-transformers

# Importing the necessary libraries
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Loading the pre-trained model
model = SentenceTransformer('all-mpnet-base-v2')

# Defining a list of sentences
sentences = [
    "Knowing the difference between similar words can be tricky",
    "It can be difficult to distinguish between words that look or sound alike",
    "The consecutive sentences check in Yoast SEO warns against repetition",
    "Avoid starting consecutive sentences with the same word to improve readability",
    "Repetition and redundancy can cause problems in writing",
    "Using the same words repeatedly can make writing boring and unengaging",
    "A simple sentence contains one independent clause with a subject and predicate",
    "A basic independent clause consists of a subject and predicate and forms a complete thought",
    "The cosine similarity measures the similarity between two vectors",
    "Cosine similarity can be used to compare the similarity between two sentences based on their vector representations"
]

# Encoding the sentences into word embeddings
word_embeddings = model.encode(sentences)
word_embeddings.shape

# Printing the word embedding of the first sentence
print(word_embeddings[0])

# Encoding the sentence to compare
sentence_to_compare = "Two sentences with different words is good for readers"
sntnc_embedding = model.encode(sentence_to_compare)

# Calculating the cosine similarity scores
scores = cos_sim(sntnc_embedding, word_embeddings[:-1])

# Finding the most similar sentence
most_similar_sentence = sentences[scores.argmax().item()]
print(most_similar_sentence)

# Creating a tuple of similarity scores and sentences
my_tuple = [(sim, value) for sim, value in zip(scores[0], sentences)]

# Sorting the tuple based on similarity scores
sorted_tuple = sorted(my_tuple, key=lambda x: x[0], reverse=True)

# Printing the ranked tuple based on similarity scores
for s in sorted_tuple:
    print(s)