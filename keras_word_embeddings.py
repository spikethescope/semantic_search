from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import numpy as np
# define documents
docs = [
    "The sun is shining brightly, casting a warm glow on the beautiful flowers",
    "I feel incredibly grateful for the support and love I receive from my friends and family",
    "The delicious aroma of freshly baked cookies fills the kitchen, making me smile",
    "I'm thrilled to have the opportunity to travel to my dream destination next week",
    "The laughter of children playing in the park brings joy to my heart",
    "The constant rain ruined my plans for a picnic in the park",
    "I'm disappointed with the poor customer service I received at the restaurant last night",
    "The long traffic jam made me late for an important meeting, causing frustration",
    "I'm feeling overwhelmed with the amount of work I have to complete before the deadline",
    "The news of the recent natural disaster saddened me deeply"
]

# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
#print(encoded_docs)
# pad documents to a max length of 4 words
max_length = np.argmax( [len(d) for d in docs])
max_length = vocab_size
print(max_length)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print("\n Padded docs ", padded_docs)
# define the model
model = Sequential()
#SET EMBEDDING DIMENSION
embed_size = 128
model.add(Embedding(input_dim = vocab_size, output_dim = embed_size, input_length=max_length))
model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs = 200, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
# print the weights of the embedding layer
weights = model.layers[0].get_weights()[0]
print(weights)
print(weights.shape)
model.save('embed_model.h5')


from keras.models import load_model
model = load_model('embed_model.h5')

# get the embedding weights
weights = model.layers[0].get_weights()[0]
print(weights.shape)

print(np.dot(padded_docs[1], weights))
#new_embed_vector = np.dot(new_padded_doc,weights)

embeddings = [np.dot(p_doc, weights) for p_doc in padded_docs]
#print(embeddings)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query = ["The recent natural disaster news filled me with profound sadness"]
#query  = ["The enticing scent of freshly baked cookies in the kitchen brought a smile to my face"]
query = ["Im frustrated to finish many works to be completed before next week"]
new_encoded_doc = [one_hot(d, vocab_size) for d in query]
print("Query One hot vector ", new_encoded_doc)

newpad = pad_sequences(new_encoded_doc, maxlen = max_length, padding = 'post')
print("\nPadded Encoded vector ", newpad)

newembed = np.dot(newpad, weights )
print("\nQuery Embedding ", newembed)

similarity = cosine_similarity(newembed, embeddings)
# print the similarity
print("\nSimilarity Scores ", similarity)

print(f"\nArgmax Index {np.argmax(similarity).item()} Matching vector from corpus is ", docs[np.argmax(similarity)])

print(docs[np.argmax(similarity)])
my_tuple = [ (sim,value) for sim, value in zip(similarity[0], docs) ]
sorted_tuple = sorted(my_tuple, key=lambda x: x[0])[::-1]
for s in sorted_tuple:
  print(s)


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


new_doc = ['You are good']
print("VOcab size", vocab_size)
new_encoded_doc = [one_hot(d, vocab_size) for d in new_doc]
print(new_encoded_doc)
# pad documents to a max length of 4 words
max_length = 20
new_padded_doc = pad_sequences(new_encoded_doc, maxlen=max_length, padding='post')
print(new_padded_doc)
print(new_padded_doc.shape)
#np.dot(new_padded_doc, weights)
print("Reshape -1 is ",new_padded_doc.reshape(-1))


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
import numpy as np
# define documents
docs = [
    "The sun is shining brightly, casting a warm glow on the beautiful flowers",
    "I feel incredibly grateful for the support and love I receive from my friends and family",
    "The delicious aroma of freshly baked cookies fills the kitchen, making me smile",
    "I'm thrilled to have the opportunity to travel to my dream destination next week",
    "The laughter of children playing in the park brings joy to my heart",
    "The constant rain ruined my plans for a picnic in the park",
    "I'm disappointed with the poor customer service I received at the restaurant last night",
    "The long traffic jam made me late for an important meeting, causing frustration",
    "I'm feeling overwhelmed with the amount of work I have to complete before the deadline",
    "The news of the recent natural disaster saddened me deeply"
]

# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
#print(encoded_docs)
# pad documents to a max length of 4 words
max_length = np.argmax( [len(d) for d in docs])
max_length = vocab_size
print(max_length)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print("\n Padded docs ", padded_docs)
# define the model
model = Sequential()
#SET EMBEDDING DIMENSION
embed_size = 128
model.add(Embedding(input_dim = vocab_size, output_dim = embed_size, input_length=max_length))
model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs = 200, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))
# print the weights of the embedding layer
weights = model.layers[0].get_weights()[0]
print(weights)
print(weights.shape)
model.save('embed_model.h5')


from keras.models import load_model
model = load_model('embed_model.h5')

# get the embedding weights
weights = model.layers[0].get_weights()[0]
print(weights.shape)

print(np.dot(padded_docs[1], weights))
#new_embed_vector = np.dot(new_padded_doc,weights)

embeddings = [np.dot(p_doc, weights) for p_doc in padded_docs]
#print(embeddings)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query = ["The recent natural disaster news filled me with profound sadness"]
#query  = ["The enticing scent of freshly baked cookies in the kitchen brought a smile to my face"]
query = ["Im frustrated to finish many works to be completed before next week"]
new_encoded_doc = [one_hot(d, vocab_size) for d in query]
print("Query One hot vector ", new_encoded_doc)

newpad = pad_sequences(new_encoded_doc, maxlen = max_length, padding = 'post')
print("\nPadded Encoded vector ", newpad)

newembed = np.dot(newpad, weights )
print("\nQuery Embedding ", newembed)

similarity = cosine_similarity(newembed, embeddings)
# print the similarity
print("\nSimilarity Scores ", similarity)

print(f"\nArgmax Index {np.argmax(similarity).item()} Matching vector from corpus is ", docs[np.argmax(similarity)])

print(docs[np.argmax(similarity)])
my_tuple = [ (sim,value) for sim, value in zip(similarity[0], docs) ]
sorted_tuple = sorted(my_tuple, key=lambda x: x[0])[::-1]
for s in sorted_tuple:
  print(s)


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


new_doc = ['You are good']
print("VOcab size", vocab_size)
new_encoded_doc = [one_hot(d, vocab_size) for d in new_doc]
print(new_encoded_doc)
# pad documents to a max length of 4 words
max_length = 20
new_padded_doc = pad_sequences(new_encoded_doc, maxlen=max_length, padding='post')
print(new_padded_doc)
print(new_padded_doc.shape)
#np.dot(new_padded_doc, weights)
print("Reshape -1 is ",new_padded_doc.reshape(-1))
