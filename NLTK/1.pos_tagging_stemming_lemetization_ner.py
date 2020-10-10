import nltk

#text = 'C. Ronaldo is a great football palyer and he loves wolves'
#text = 'Albert Enistine was born in 1879'
text = 'Steve Jobs was the CEO of Apple Corporation located in America'
print(text)

## tokenization ##
from nltk.tokenize import word_tokenize
tokenize_text = word_tokenize(text)
print('Tokenize Text : {}'.format(tokenize_text))

## part od speech (pos) tagging ##
from nltk import pos_tag
pos_text = pos_tag(tokenize_text)
print('POS Tagging : {}'.format(pos_text))

## stemming ##
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
stem_word = []
for word in tokenize_text:
	stem_word.append(porter_stemmer.stem(word))
print('Stem Word : {}'.format(stem_word))

## lemmatization ##
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemma_word = []
for word in tokenize_text:
	lemma_word.append(lemmatizer.lemmatize(word))
print('Lemma Word : {}'.format(lemma_word))

## named entity recognition (NER) ##
from nltk import ne_chunk
ner = ne_chunk(pos_text)
print('NER : {}'.format(ner))
ner.draw()
