import re
import time
from collections import defaultdict

import nltk
from nltk.corpus import stopwords


# nltk.download( 'stopwords' )

class Graph:
    # Constructor
    def __init__(self):
        # default dictionary to store graph
        self.graph = defaultdict( dict )

    def addEdge(self, u, v, w):
        self.graph[u] = {v: w}


def corpus_reader(param):
    pattern = '((?<=\(\')[^\']*(?=\',)|(?<=\(\")[^\"]*(?=\",))'
    sentences = []
    with open( param ) as f:
        corpus = f.readlines()
    for line in corpus:
        temp = re.findall( pattern, line )
        sentences.append( temp )

    return sentences


def collocate(observed_sentences, param):
    col_dict = dict()
    max_co_occurence = 0
    for sentence in observed_sentences:
        for itr in range( len( sentence ) ):
            frwd_candidates = [i + itr for i in range( 1, param + 1 )]
            bkwd_candidates = [itr - i for i in range( 1, param + 1 )]

            centre = sentence[itr]
            for can in frwd_candidates:
                try:
                    x = sentence[int( can )]
                    if (x, centre) in col_dict.keys():
                        col_dict[(x, centre)] += 1
                        max_co_occurence = max_co_occurence if max_co_occurence > col_dict[(x, centre)] else col_dict[
                            (x, centre)]
                    else:
                        col_dict[(x, centre)] = 1
                except IndexError:
                    pass

            for can in bkwd_candidates:
                try:
                    x = sentence[int( can )]
                    if (x, centre) in col_dict.keys():
                        col_dict[(x, centre)] += 1
                        max_co_occurence = max_co_occurence if max_co_occurence > col_dict[(x, centre)] else col_dict[
                            (x, centre)]
                    else:
                        col_dict[(x, centre)] = 1
                except IndexError:
                    pass

    return col_dict, max_co_occurence


if __name__ == "__main__":
    start_time = time.time()
    observed_sentences = corpus_reader( "./corpus.txt" )

    stop_words = set( stopwords.words( 'english' ) )

    # the sentences are filtered to remove all stop words, non-alphabetic tokens while also allowing hyphenated words
    for itr, sentence in enumerate( observed_sentences ):
        filtered_sentence = [w for w in sentence if not w in stop_words and (w.isalpha() or (
                len( w.split( '-' ) ) == 2 and w.split( '-' )[0].isalpha() and w.split( '-' )[1].isalpha()))]
        observed_sentences[itr] = filtered_sentence

    # corpus filtered and ready

    collocations_dict, max_co_occurance = collocate( observed_sentences, 3 )

    # collocations ready

    G = Graph()
    for (w1, w2), val in collocations_dict.items():
        G.addEdge( w1, w2, val )
        G.addEdge( w2, w1, val )

    focus_word=input(" Enter the focus word :- ")
    # Todo: what the fuck is after this?