import nltk
from nltk.book import *
from nltk.corpus import movie_reviews, stopwords, words

import string

def question2():
    # Consider text1 ('Moby Dick'), text2 ('Sense and Sensibility'), text3 ('Book of Genesis'), text7('Wall Street Journal')

    # Write a loop that displays, for each text, which words are similar to:

    #     'great',
    #     'king',
    #     'country',
    #     'fear',
    #     'love'


    similar_words_targets = ['great', 'king', 'country', 'fear', 'love']

    all_texts = []

    TEXTBOOK_PREFIX = 'text'
    TEXTBOOK_SUFFIXES = [1, 2, 3, 7]

    for propertyName in dir(nltk.book):
        if propertyName.startswith(TEXTBOOK_PREFIX) and propertyName[-1].isdigit() and int(propertyName[-1]) in TEXTBOOK_SUFFIXES:
            all_texts.append(getattr(nltk.book, propertyName))

    for currWord in similar_words_targets:
        print("-"*10 + "\n")
        print("Curr Word Target -> %s" %(currWord))
        for currText in all_texts:
            currText.similar(currWord)
            print("\n")
        print("-"*10 + "\n")


    # Write a loop that, for each text, generates a 50-token random sequence.

    ## Question, is this it?
    for i in all_texts:
        i.generate(50)

    # Now let's make two Frequency Distributions.
    # We'll use the movie_reviews corpus.
    # Construct one Frequency Distribution that counts all of the words in the positive reviews, and one that counts all of the words in the negative reviews.
    # Print the 10 most common words in each distribution.

    pos_freq_dist = nltk.FreqDist(movie_reviews.words(categories='pos'))
    neg_freq_dist = nltk.FreqDist(movie_reviews.words(categories='neg'))

    print("Positive Reviews")
    print(pos_freq_dist.most_common(10))
    print("Negative Reviews")
    print(neg_freq_dist.most_common(10))


    # There's a lot of noise in our data. As in assignment1,
    # update your code to remove stopwords, non-words, and convert everything to
    # lower case. Does that help to distinguish between positive and negative reviews?

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    english_words = set(words.words())

    def clean_text(text):
        text = text.lower()
        text = ''.join([ch for ch in text if ch not in punctuation])
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([word for word in text.split() if word in english_words])
        return text

    pos_freq_dist = nltk.FreqDist([clean_text(word) for word in movie_reviews.words(categories='pos')])
    neg_freq_dist = nltk.FreqDist([clean_text(word) for word in movie_reviews.words(categories='neg')])

    pos_freq_dist.pop('')
    neg_freq_dist.pop('')

    print("Positive Reviews")
    print(pos_freq_dist.most_common(10))

    print("Negative Reviews")
    print(neg_freq_dist.most_common(10))


    # Rather than keeping our distributions in two separate data structures, let's use a ConditionalFrequencyDistribution. (see Chapter 3)
    # Add a loop that iterates through the ConditionalFrequencyDistribution and prints the 10 most common words for each category.

    cfd = nltk.ConditionalFreqDist(
        (category, word)
        for category in movie_reviews.categories()
        for word in [clean_text(word) for word in movie_reviews.words(categories=category)])

    for category in movie_reviews.categories():
        print("Category: %s" %(category))
        print(cfd[category].most_common(11))

# def main():
#     question2()

if __name__ == "__main__":
    question2()