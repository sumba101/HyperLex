import re
import time

import nltk
from nltk.corpus import stopwords


# nltk.download( 'stopwords' )

class Graph_temp:
    # Constructor
    def __init__(self):
        # default dictionary to store graph
        self.graph = dict()

    def addEdge(self, u, v, w):
        if u not in self.graph.keys():
            self.graph[u] = {v: w}
        else:
            self.graph[u].update( {v: w} )


class Graph:

    def __init__(self, vertices):
        self.V = vertices
        self.edges = []
        self.graph = dict()

    def addEdge(self, u, v, w):
        self.edges.append( [u, v, w] )

    def addGraph(self, u, v, w):
        if u not in self.graph.keys():
            self.graph[u] = {v: w}
        else:
            self.graph[u].update( {v: w} )

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find( parent, parent[i] )

    def union(self, parent, rank, x, y):
        xroot = self.find( parent, x )
        yroot = self.find( parent, y )

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def KruskalMST(self, vertices):
        result = []
        i = 0
        e = 0

        self.edges = sorted( self.edges, key=lambda item: item[2] )
        parent = dict()
        rank = dict()

        for node in vertices:
            parent[node] = node
            rank[node] = 0

        for u, v, w in self.edges:

            x = self.find( parent, u )
            y = self.find( parent, v )

            if x != y:

                result.append( [u, v, w] )
                self.union( parent, rank, x, y )

        return result


def corpus_reader(param):
    pattern = '((?<=\(\')[^\']*(?=\',)|(?<=\(\")[^\"]*(?=\",))'
    sentences = []
    with open( param ) as f:
        corpus = f.readlines()
    for line in corpus:
        temp = re.findall( pattern, line )
        sentences.append( temp )

    return sentences


def collocate(observed_sentences, param, word_freq, focus_word):
    col_dict = dict()

    for sentence in observed_sentences:
        for itr in range( len( sentence ) ):


            frwd_candidates = [i + itr for i in range( 1, param + 1 )]
            bkwd_candidates = [itr - i for i in range( 1, param + 1 )]
            centre = sentence[itr]

            for can in frwd_candidates:

                if (int( can ) < len( sentence )):

                    x = sentence[int( can )]
                    if x == centre:  # to evade collocations with same word
                        continue

                    try:
                        word_freq[x] += 1
                    except KeyError:
                        word_freq[x] = 1

                    if (x, centre) in col_dict.keys():
                        col_dict[(x, centre)] += 1
                    else:
                        col_dict[(x, centre)] = 1

            for can in bkwd_candidates:
                if (int( can ) >= 0):

                    x = sentence[int( can )]

                    if x == centre:  # to evade collocations with same word
                        continue

                    try:
                        word_freq[x] += 1
                    except KeyError:
                        word_freq[x] = 1

                    if (x, centre) in col_dict.keys():
                        col_dict[(x, centre)] += 1

                    else:
                        col_dict[(x, centre)] = 1

    return col_dict


def goodcandidate(G_mst, v):
    threshold_weight = 0.8
    #     checking for 3 neighbours
    E = []
    for neigh, val in G_mst.graph[v].items():
        E.append( [neigh, val] )
    #     using smallest 3 weights
    E = sorted( E, key=lambda x: x[1] )

    #     edges sorted on basis of edge weight
    mean = (E[0][1] + E[1][1] + E[2][1]) / 3  # taking the mean of the weights of three weights
    return mean < threshold_weight


def roothubs(G_mst, word_freq, collocations_dict, focus_word):
    threshold = 2
    H = list()

    Vertices = list()  # this is all the words that occur in collocation pair with the target word
    for w in word_freq.keys():
        if ((w, focus_word) in collocations_dict.keys()) or ((focus_word, w) in collocations_dict.keys()):
            Vertices.append( w )

    while len( Vertices ) > 0:
        temp = -1
        try:
            temp = collocations_dict[(focus_word, Vertices[0])]
        except KeyError:
            temp = collocations_dict[(Vertices[0], focus_word)]

        if temp < threshold:
            Vertices.remove( Vertices[0] )
            continue

        v = Vertices[0]

        if goodcandidate( G_mst, v ):
            H.append( v )

            for neigh, val in G_mst.graph[v].items():
                if neigh in Vertices:
                    Vertices.remove( neigh )
        #         remove the neighbours from the vertices
        Vertices.remove( v )

    return H


def helper(Resultant_mst, head, temp, focus_word):
    for l in Resultant_mst:

        if (head in l) and len( temp ) < 3:
            node = l[0] if l[1] == head else l[1]
            if node == focus_word:
                continue
            temp.append( node )

    return temp


def answer(Resultant_mst, roothubs , focus_word):
    Ans = list()  # this will be a list of 3 sense word tuple

    for hub in roothubs:
        temp = list()
        head = hub
        while len( temp ) < 3:  # till we get three sense words
            temp = helper( Resultant_mst, head, temp , focus_word)
            head=temp[0]
        Ans.append(tuple(temp))

    return Ans

if __name__ == "__main__":
    start_time = time.time()
    observed_sentences = corpus_reader( "./corpus.txt" )

    stop_words = set( stopwords.words( 'english' ) )

    # the sentences are filtered to remove all stop words, non-alphabetic tokens while also allowing hyphenated words
    for itr, sentence in enumerate( observed_sentences ):
        filtered_sentence = [w.lower() for w in sentence if (w.isalpha() or (
                len( w.split( '-' ) ) == 2 and w.split( '-' )[0].isalpha() and w.split( '-' )[1].isalpha()))]

        filtered_sentence = [w for w in filtered_sentence if w not in stop_words]

        observed_sentences[itr] = filtered_sentence

    # corpus filtered and ready

    focus_word = input()

    target_sentences = []
    # target sentences has only the sentences that have the focus word in it
    for sent in observed_sentences:
        if focus_word in sent:
            target_sentences.append( sent )

    word_freq = dict()

    collocations_dict = collocate( target_sentences, 3, word_freq, focus_word )
    # collocations ready

    # order the word frequencies in decending order
    word_freq = {k: v for k, v in sorted( word_freq.items(), reverse=True, key=lambda item: item[1] )}

    # The word frequency key has all the words + the focus word
    # we can get the words that occur in collocation pair witht he focus word from the collocations dictionary

    nodes_num = len( word_freq.keys() ) - 1  # The -1 is to not count the target word

    G_mst = Graph( nodes_num )

    for (w1, w2), val in collocations_dict.items():
        if w1 == focus_word or w2 == focus_word:
            continue
        # Making the graph from words that aren't the focus word
        maximum = (val / word_freq[w1]) if (val / word_freq[w1]) > (val / word_freq[w2]) else (val / word_freq[w2])
        edge_weight = 1 - maximum
        # making the edges
        G_mst.addEdge( w1, w2, edge_weight )

        G_mst.addGraph( w1, w2, edge_weight )
        G_mst.addGraph( w2, w1, edge_weight )

    hubs = roothubs( G_mst, word_freq, collocations_dict, focus_word )

    G_mst.V += 1  # For the focus word addition

    for node in hubs:
        G_mst.addEdge( focus_word, node, 0 )
        G_mst.addGraph( focus_word, node, 0 )

    vertices = list( word_freq.keys() )
    vertices.append( focus_word )

    Resultant_mst = G_mst.KruskalMST( vertices )
    Answer = answer( Resultant_mst, hubs ,focus_word)

    for i,a in enumerate(Answer):
        if i != 0:
            print(',',end='')
        print(a,end="")
