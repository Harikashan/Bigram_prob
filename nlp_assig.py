import nltk
import pandas as pd
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def removePunctuation(collection):
    removedPunc = collection['body'].str.replace(r'[^\w\s]+', '')
    return  removedPunc


def toLowercase(collection):
    lowercased = collection['punctRemoved'].str.lower()
    return lowercased

def removeStopWord(collection):
    stop = stopwords.words('english')
    removedStop = collection['lowerCased'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return removedStop

def wordTokenize(collection):
    tokenizedWord = collection['stopRemoved'].apply(nltk.word_tokenize)
    return tokenizedWord

def wordnetLemmatize(collection):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatizedWord = collection['wordTokenized'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])
    return lemmatizedWord

def createBigram(collection):
    bigram = collection['lemmatizedWord'].apply(lambda x: list(ngrams(x,2)))
    return bigram

def create_bigram_dictionary(bigram_corpus):
    bigram_dic = {}
    for bigram in bigram_corpus:
        if bigram not in bigram_dic:
            bigram_dic[bigram] = 1
        else:
            bigram_dic[bigram] += 1
    return bigram_dic

def create_unigram_dictionary(unigram_corpus):
    unigram_dic = {}
    for unigram in unigram_corpus:
        if unigram not in unigram_dic:
            unigram_dic[unigram] = 1
        else:
            unigram_dic[unigram] += 1
    # print(len(unigram_dic))
    return unigram_dic

def calculate_ham_bigram_probability(w1, w2):
    try:
        no_w1 = ham_unigram_dictionary[w1]
    except KeyError:
        no_w1 = 0

    try:
        w1_w2 = ham_bigram_dictionary[w1, w2]
    except KeyError:
        w1_w2 = 0

    V = len(ham_unigram_dictionary)
    p = (w1_w2 + 1) / (no_w1 + V)
    return p


def calculate_spam_bigram_probability(w1, w2):
    try:
        no_w1 = spam_unigram_dictionary[w1]
    except KeyError:
        no_w1 = 0

    try:
        w1_w2 = spam_bigram_dictionary[w1, w2]
    except KeyError:
        w1_w2 = 0

    V = len(spam_unigram_dictionary)
    p = (w1_w2 + 1) / (no_w1 + V)
    return p

def calculate_ham_laplace_smoothing_prob(bigrams):
    p = 1
    for bigram in bigrams:
        p = p * calculate_ham_bigram_probability(bigram[0], bigram[1])
    return p

def calculate_spam_laplace_smoothing_prob(bigrams):
    p = 1
    for bigram in bigrams:
        p = p * calculate_spam_bigram_probability(bigram[0], bigram[1])
    return p


def sentancePreprocessing(message):
    message = message.lower()
    message = re.sub(r'[^\w\s]', '', message)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(message)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    message = " ".join(filtered_sentence)
    word_tokens = word_tokenize(message)

    # StemmedWords = []
    # for ms in word_tokens:
    #     StemmedWords.append(PorterStemmer().stem(ms))

    LemmatizedWords = []
    for ms in word_tokens:
        LemmatizedWords.append(WordNetLemmatizer().lemmatize(ms))

    bigram = list(ngrams(LemmatizedWords, 2))
    bigram_d = create_bigram_dictionary(bigram)
    # print(bigram_d)
    return bigram


collection = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)
collection.columns = ["label", "body"]
collection.head()

collection['punctRemoved'] = removePunctuation(collection)
collection['lowerCased'] = toLowercase(collection)
collection['stopRemoved'] = removeStopWord(collection)
collection['wordTokenized'] = wordTokenize(collection)
# collection['stemmedWord'] = wordStemming(collection)
collection['lemmatizedWord'] = wordnetLemmatize(collection)
collection['bigrams'] = createBigram(collection)



collection['bigrams'] = collection['lemmatizedWord'].apply(lambda x: list(ngrams(x, 2)))
bigram_data_frame = collection.groupby('label').agg({'bigrams': 'sum'})
unigram_data_frame = collection.groupby('label').agg({'lemmatizedWord': 'sum'})

ham_unigram_collection = unigram_data_frame.iat[0, 0]
spam_unigram_collection = unigram_data_frame.iat[1, 0]


ham_bigram_collection = bigram_data_frame.iat[0, 0]
spam_bigram_collection = bigram_data_frame.iat[1, 0]

spam_unigram_dictionary = create_unigram_dictionary(spam_unigram_collection)
ham_unigram_dictionary = create_unigram_dictionary(ham_unigram_collection)

spam_bigram_dictionary= create_bigram_dictionary(spam_bigram_collection)
ham_bigram_dictionary= create_bigram_dictionary(ham_bigram_collection)

# print(calculate_spam_bigram_probability('ive', 'search'))
# Corpus.to_csv('out1.csv')
# print(spam_bigram_dictionary)

msg1 = 'Sorry, ..use your brain dear'
msg2 = 'SIX chances to win CASH.'

# msg2 = 'ive been searching for good job'

ham_msg1 = calculate_ham_laplace_smoothing_prob(sentancePreprocessing(msg1))
spam_msg1 = calculate_spam_laplace_smoothing_prob(sentancePreprocessing(msg1))

ham_msg2 = calculate_ham_laplace_smoothing_prob(sentancePreprocessing(msg2))
spam_msg2 = calculate_spam_laplace_smoothing_prob(sentancePreprocessing(msg2))

print(ham_msg1,spam_msg1, ham_msg2, spam_msg2)

print("Message 1 = 'Sorry, ..use your brain dear'")
print("ham_prob = ", ham_msg1)
print("spam_prob = ", spam_msg1)

if ham_msg1>spam_msg1:
    print('ham')

else:
    print('spam')


print("\nMessage 2 = 'SIX chances to win CASH.'")
print("ham_prob = ", ham_msg2)
print("spam_prob = ", spam_msg2)

if ham_msg2>spam_msg2:
    print('ham')

else:
    print('spam')