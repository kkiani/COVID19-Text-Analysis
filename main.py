__author__  =   "Kiarash Kiani"
__email__   =   "hi@kiarash.info"

from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from hazm import word_tokenize, stopwords_list
from nltk.corpus import PlaintextCorpusReader
import pandas as pd
import emoji
from multiprocessing import Pool
import tqdm
import time

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)

def remove_stopwords(doc):
    return [word for word in doc if not word in STOP_WORDLIST]

def load_stopwords():
    with open('data/stopwords.txt') as word_file:
        return word_file.read().split('\n')

def main():
    print('- loading dataset')
    df = pd.read_csv('data/Labeled-Data-v1.csv')
    docs = df['Content'].values
    docs = [remove_emoji(doc) for doc in docs]      # removing emojies

    print('- tokenize documents')
    tokenized_docs = [word_tokenize(doc) for doc in docs]

    print('- cleaning data')
    cleaned_data = []
    with Pool() as pool:
        cleaned_data = list(tqdm.tqdm(pool.imap(remove_stopwords, tokenized_docs), total=len(tokenized_docs)))

    print('- Create a corpus from a list of texts')
    common_dictionary = Dictionary(cleaned_data)
    common_corpus = [common_dictionary.doc2bow(text) for text in cleaned_data]

    print('- Train the model on the corpus.')
    lda = LdaModel(common_corpus, num_topics=20, id2word = common_dictionary, passes = 100, alpha='auto', update_every=5)

    for idx, topic in lda.show_topics(formatted=False, num_words= 4):
        print('Topic: {} \nWords: {}'.format(idx, '|'.join([w[0] for w in topic])))



if __name__ == "__main__":

    STOP_WORDLIST = load_stopwords()

    print(f'hazm stopwords: {len(stopwords_list())}, my stopwords: {len(STOP_WORDLIST)}')

    # main()
    # print(load_stopwords())