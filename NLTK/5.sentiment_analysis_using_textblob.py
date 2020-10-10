from textblob import TextBlob

feedback1 = 'The food at Radisson was awesome'
feedback2 = 'The food at Radisson was very good'
feedback3 = 'The food at Radisson was not good'

blob1 = TextBlob(feedback1)
blob2 = TextBlob(feedback2)
blob3 = TextBlob(feedback3)

print(blob1.sentiment)
print(blob2.sentiment)
print(blob3.sentiment)

# Polarity -> It simply means emotions expressed in a sentence. Emotions are closely related to sentiments. E.g. 'This car is worth the price'
# Subjectivity -> Subjective sentence expresses some personal feelings, views, or beliefs. E.g. 'I like iPhone'