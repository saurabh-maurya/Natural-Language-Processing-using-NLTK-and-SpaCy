import nltk

sample_text = ["The quick brown fox jumped over the lazy dog.",
				"The dog.",
				"The fox"			
			]
print(sample_text)

## Frequency distribution ##
from nltk import FreqDist
ferquency_distribution = FreqDist(sample_text) # number of occurence of words in given text
unique_words_count = len(ferquency_distribution)
print(unique_words_count)
n_most_common_words = ferquency_distribution.most_common(5) # here n = 5
print(n_most_common_words)

## Count Vectorizer ##
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(sample_text)
print(vectorizer.vocabulary_)
vector = vectorizer.transform(sample_text)
print(vector.shape)
print(type(vector))
print(vector.toarray())

## tf-idf Vectorizer ##
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(sample_text)
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
vector = vectorizer.transform([sample_text[0]])
print(vector.shape)
print(vector.toarray())