import re
import numpy as np
import pickle
import nltk
import builtins
from nltk.corpus import stopwords
import json
from contextlib import closing
from nltk.stem import *
from collections import defaultdict, Counter
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from inverted_index_gcp import InvertedIndex, MultiFileReader
import math
from google.cloud import storage
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from collections import defaultdict, Counter
import concurrent.futures
from sklearn.preprocessing import MinMaxScaler
import gensim.models
import gensim.downloader as api
from time import time
model_glove_wiki = api.load("glove-wiki-gigaword-300")
print('Done loading w2v')

client = storage.Client()
bucket = client.get_bucket('full_indexex_020324_yuval')
bucket_name = 'full_indexex_020324_yuval'

def read_pickle(bucket_name, pickle_route):
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(pickle_route)
    pick = pickle.loads(blob.download_as_string())
    return pick


# -----------init-indexes------------------#
print('Start loading Indexes')
body_index = read_pickle("full_indexex_020324_yuval","body_index_tokenize_stem/index_body_index_tokenize_stem.pkl")
body_index.true_location = "body_index_tokenize_stem"

title_index_bigram = read_pickle("full_indexex_020324_yuval","title_index_bigram_stem_new/index_title_index_bigram_stem_new.pkl")
title_index_bigram.true_location = "title_index_bigram_stem_new"

title_index = read_pickle("full_indexex_020324_yuval","title_index_stem_new/index_title_index_stem_new.pkl")
title_index.true_location = "title_index_stem_new"

page_rank = read_pickle("all_indexes_final_project", "pagerank_dict.pkl")

page_views = read_pickle("all_indexes_final_project", "pageviews.pkl")

id2title = read_pickle("all_indexes_final_project", "id_to_title_dict.pkl")


# ------------helper-functions---------#

def is_number(s):
    try:
        float(s)  # for float and int representations
        return True
    except ValueError:
        return False


def similar_words(list_of_tokens,ecc=0.7):
    """
    Finds words similar to the given list of tokens using a Word2Vec model and combines them with the original tokens.
    
    Parameters:
    - list_of_tokens: List of words/tokens to find similarities for.
    - model: Word2Vec model used for finding similar words.
    - ecc: Similarity threshold; only words with a similarity above this threshold are considered.

    Returns:
    - A list containing the original tokens and the similar words found.
    """
    try:
        candidates = model_glove_wiki.most_similar(positive=list_of_tokens, topn=4)
    except Exception as e:
      return list_of_tokens
    res = [word for word, similarity in candidates if similarity > ecc]
    
    if len(res) > 2:
        # If more than two similar words are found, print 'gk' and add each word to the list.
        list_of_tokens.extend(res) # Changed to extend to add all items in res.
    elif len(res) <= 2 and res: # Ensures res is not empty before extending the list.
        list_of_tokens.extend(res)
        
    return list_of_tokens


#------Tokenizer--------#
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became","considered"]

all_stopwords = set([word.lower() for word in english_stopwords.union(corpus_stopwords)])

ps = PorterStemmer()

token_pattern = re.compile(r"""[\w](['\-]?\w)*""")

def get_tokens(text, stem=True, bigrams=False, trigrams=False,w2v=False):
    # Tokenize: Extract tokens using regular expression
    tokens = [token.group() for token in token_pattern.finditer(text.lower())]
    
    # Filter out stop words from either stemmed or original tokens
    filtered_tokens = [token for token in tokens if token not in all_stopwords]
    # print('before w2v:', filtered_tokens)
    if w2v:
        if not any([is_number(s) for s in filtered_tokens]):
            try:
                new_tokens = similar_words(filtered_tokens)
                filtered_tokens =   new_tokens
                # print('after w2v:', new_tokens)
            except Exception as e:
                  print(e)
    
    # Apply stemming if requested
    if stem:
        filtered_tokens = [ps.stem(filtered_tokens) for filtered_tokens in filtered_tokens]
        
    # Generate bigrams or trigrams if requested
    if bigrams:
        filtered_tokens = [f"{filtered_tokens[i]} {filtered_tokens[i+1]}" for i in range(len(filtered_tokens)-1)]
    if trigrams:
        filtered_tokens = [f"{filtered_tokens[i]} {filtered_tokens[i+1]} {filtered_tokens[i+2]}" for i in range(len(filtered_tokens)-2)]
    
    return filtered_tokens

def retrieve_top_documents(similarity_scores, top_n=300):
    """
    Retrieves the top N documents based on their similarity scores.
    
    Parameters:
    - similarity_scores (dict): A dictionary where keys are document IDs and values are similarity scores.
    - top_n (int): The number of top documents to retrieve based on their scores.
    
    Returns:
    - list: A list of tuples containing the document ID and its score, sorted by score in descending order.
    """
    return sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def find_relevant_documents(query_tokens, inverted_index, base_directory=''):
    """
    Retrieves relevant documents for a given query along with term frequencies from an inverted index.
    
    Parameters:
    - query_tokens (list of str): Tokens extracted from the query.
    - inverted_index (InvertedIndex): The inverted index object containing document frequencies and posting lists.
    - base_directory (str, optional): Base directory for reading posting lists. Defaults to an empty string.
    
    Returns:
    - set: Unique document IDs relevant to the query.
    - dict: Mapping of query tokens to another dict, mapping document IDs to term frequencies.
    """
    unique_tokens = np.unique(query_tokens) 
    document_frequencies = inverted_index.df  
    relevant_docs = set()
    term_to_doc_freqs = {}

    for token in unique_tokens:
        if token in document_frequencies:
            posting_list = inverted_index.read_a_posting_list(token, base_dir=base_directory, bucket_name=bucket_name)
            doc_ids_with_freq = dict(posting_list)
            term_to_doc_freqs[token] = doc_ids_with_freq
            relevant_docs.update(doc_ids_with_freq.keys())

    return relevant_docs, term_to_doc_freqs

#------BM25CLASS-------#
class BM25_from_index:

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        # print(index.document_len)
        self.N = len(index.document_len)
        self.AVGDL = builtins.sum(index.document_len.values()) / self.N

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def _score(self, query, doc_id, candidate_dict):
        score = 0.0
        doc_len = self.index.document_len[doc_id]
        for term in query:
            if term in self.index.df:
                term_frequencies = candidate_dict[term]
                if doc_id in term_frequencies:
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score

    def search(self, query_tokens, top_n=100):
        """
        Performs a search to find the top N documents matching the query tokens.

        Parameters:
        - query_tokens (list of str): The query tokens.
        - top_n (int): The number of top documents to return.

        Returns:
        - list: A list of tuples containing the document ID and its score, sorted by score in descending order.
        """
        relevant_docs, doc_frequencies = find_relevant_documents(query_tokens, self.index, base_directory='')
        self.idf = self.calc_idf(query_tokens)
        temp_scores = {doc_id: self._score(query_tokens, doc_id, doc_frequencies) for doc_id in relevant_docs}
        top_documents = retrieve_top_documents(temp_scores, top_n)
        return top_documents


def BM25_search(index, tokens, results_count=100):
    return BM25_from_index(index).search(tokens, results_count)

def compute_unique_term_scores(row, ngram_cands):
    """
    Adapted function to compute scores for a single row (document) based on the ngram candidates.
    
    Parameters:
    - row (pandas.Series): A row from the df_scores DataFrame.
    - ngram_cands (dict): Pre-filtered mapping from query bigrams to documents and their term frequencies.
    
    Returns:
    - score (int): The number of unique bigram terms from the query that appear in the document.
    """
    doc_id = row['doc_id']
    score = 0
    for doc_dict in ngram_cands.values():
        if doc_id in doc_dict:
            score += 1
    return score

def get_page_rank_score(row):
    doc_id = row['doc_id']
    # Assuming page_rank is a dictionary with doc_id as keys
    return page_rank.get(doc_id, 0)

def get_page_views_score(row):
    doc_id = row['doc_id']
    # Assuming page_views is a dictionary with doc_id as keys
    return page_views.get(doc_id, 0)

def information_retrieval(query, index_title, index_title_bigram, index_body, weights=[1, 1, 1, 3, 3], N=100, debug_mode=False):
    timings = {}
    start_time = time()

    # Step 1: Tokenize the query
    query_tokens = get_tokens(query, stem=True)
    query_tokens_bigrams = get_tokens(query, stem=True, bigrams=True)
    if debug_mode:
        timings['tokenization'] = time() - start_time

    # BM25 searches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        start_time = time()
        future_body = executor.submit(BM25_search, index_body, query_tokens, N)
        future_title = executor.submit(BM25_search, index_title, query_tokens, N)
        
        bm25_body_cands = future_body.result()
        timings['bm25_body_search'] = time() - start_time

        start_time = time()
        bm25_title_cands = future_title.result()
        timings['bm25_title_search'] = time() - start_time

    # Processing and merging results
    df_body = pd.DataFrame(bm25_body_cands, columns=['doc_id', 'bm25_body_score'])
    df_title = pd.DataFrame(bm25_title_cands, columns=['doc_id', 'bm25_title_score'])
    df_scores = pd.merge(df_body, df_title, on='doc_id', how='outer').fillna(0)

    # Bigram score retrieval
    start_time = time()
    ngram_cands = find_relevant_documents(query_tokens_bigrams, index_title_bigram)[1]
    filtered_ngram_cands = {term: doc_dict for term, doc_dict in ngram_cands.items() if any(doc_id in df_scores['doc_id'].values for doc_id in doc_dict)}
    df_scores['score_ngram'] = df_scores.apply(compute_unique_term_scores, axis=1, args=(filtered_ngram_cands,))
    if debug_mode:
        timings['ngram_search'] = time() - start_time

    # Page rank and page view score retrieval
    start_time = time()
    df_scores['page_rank_score'] = df_scores.apply(get_page_rank_score, axis=1)
    df_scores['page_views_score'] = df_scores.apply(get_page_views_score, axis=1)
    if debug_mode:
        timings['page_rank_views'] = time() - start_time

    # Normalization and final scoring
    start_time = time()
    score_columns = ['bm25_body_score', 'bm25_title_score', 'score_ngram', 'page_rank_score', 'page_views_score']
    normalized_columns = [col + '_normalized' for col in score_columns]
    scaler = MinMaxScaler()
    df_scores[normalized_columns] = scaler.fit_transform(df_scores[score_columns])
    df_scores['final_score'] = df_scores[normalized_columns].mul(weights).sum(axis=1)
    top_n_documents = df_scores.sort_values(by='final_score', ascending=False).head(N)
    if debug_mode:
        timings['scoring_normalization'] = time() - start_time

    # Extracting top N document IDs and titles
    top_n_doc_ids = top_n_documents['doc_id'].values
    res = [(str(doc_id), id2title[doc_id]) for doc_id in top_n_doc_ids]

    if debug_mode:
        return res, timings
    return res


def only_body_bm25(query, index_title, index_title_bigram, index_body, weights=[1, 1, 1, 3, 3], N=100, debug_mode=True):
    timings = {}
    start_time = time()

    # Step 1: Tokenize the query
    query_tokens = get_tokens(query, stem=True)
    # query_tokens_bigrams = get_tokens(query, stem=True, bigrams=True)
    if debug_mode:
        timings['tokenization'] = time() - start_time

    # BM25 searches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        start_time = time()
        future_body = executor.submit(BM25_search, index_body, query_tokens, N)
        
        bm25_body_cands = future_body.result()
        timings['bm25_body_search'] = time() - start_time

    # Processing and merging results
    df_body = pd.DataFrame(bm25_body_cands, columns=['doc_id', 'bm25_body_score'])

    top_n_documents = df_body.sort_values(by='bm25_body_score', ascending=False).head(N)

    # Extracting top N document IDs and titles
    top_n_doc_ids = top_n_documents['doc_id'].values
    res = [(doc_id, id2title[doc_id]) for doc_id in top_n_doc_ids]

    if debug_mode:
        return res, timings
    return res


def only_title(query, index_title, index_title_bigram, index_body, weights=[1, 1, 1, 3, 3], N=100, debug_mode=True):
    timings = {}
    start_time = time()
    query_tokens = get_tokens(query, stem=True)

    # BM25 searches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        start_time = time()
        future_title = executor.submit(BM25_search, index_title, query_tokens, N)
        bm25_title_cands = future_title.result()
        timings['bm25_title_search'] = time() - start_time


    df_title = pd.DataFrame(bm25_title_cands, columns=['doc_id', 'bm25_title_score'])

    top_n_documents = df_title.sort_values(by='bm25_title_score', ascending=False).head(N)

    # Extracting top N document IDs and titles
    top_n_doc_ids = top_n_documents['doc_id'].values
    res = [(doc_id, id2title[doc_id]) for doc_id in top_n_doc_ids]

    if debug_mode:
        return res, timings
    return res

def page_rank_and_titles(query, index_title, index_title_bigram, index_body, weights=[1, 1, 3, 3], N=100, debug_mode=True):
    timings = {}
    start_time = time()

    # Step 1: Tokenize the query
    query_tokens = get_tokens(query, stem=True)
    query_tokens_bigrams = get_tokens(query, stem=True, bigrams=True)
    if debug_mode:
        timings['tokenization'] = time() - start_time

    # BM25 searches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        start_time = time()
        future_title = executor.submit(BM25_search, index_title, query_tokens, N)
        bm25_title_cands = future_title.result()
        timings['bm25_title_search'] = time() - start_time

    df_title = pd.DataFrame(bm25_title_cands, columns=['doc_id', 'bm25_title_score'])

    # Bigram score retrieval
    start_time = time()
    ngram_cands = find_relevant_documents(query_tokens_bigrams, index_title_bigram)[1]
    filtered_ngram_cands = {term: doc_dict for term, doc_dict in ngram_cands.items() if any(doc_id in df_title['doc_id'].values for doc_id in doc_dict)}
    df_title['score_ngram'] = df_title.apply(compute_unique_term_scores, axis=1, args=(filtered_ngram_cands,))
    if debug_mode:
        timings['ngram_search'] = time() - start_time

    # Page rank and page view score retrieval
    start_time = time()
    df_title['page_rank_score'] = df_title.apply(get_page_rank_score, axis=1)
    df_title['page_views_score'] = df_title.apply(get_page_views_score, axis=1)
    if debug_mode:
        timings['page_rank_views'] = time() - start_time

    # Normalization and final scoring
    start_time = time()
    score_columns = ['bm25_title_score', 'score_ngram', 'page_rank_score', 'page_views_score']
    normalized_columns = [col + '_normalized' for col in score_columns]
    scaler = MinMaxScaler()
    df_title[normalized_columns] = scaler.fit_transform(df_scores[score_columns])
    df_title['final_score'] = df_title[normalized_columns].mul(weights).sum(axis=1)
    top_n_documents = df_title.sort_values(by='final_score', ascending=False).head(N)
    if debug_mode:
        timings['scoring_normalization'] = time() - start_time

    # Extracting top N document IDs and titles
    top_n_doc_ids = top_n_documents['doc_id'].values
    res = [(doc_id, id2title[doc_id]) for doc_id in top_n_doc_ids]

    if debug_mode:
        return res, timings
    return res

def final_with_w2c(query, index_title, index_title_bigram, index_body, weights=[1, 1, 1, 3, 3], N=100, debug_mode=True):
    timings = {}
    start_time = time()

    # Step 1: Tokenize the query
    query_tokens = get_tokens(query, stem=True,w2v=True)
    query_tokens_bigrams = get_tokens(query, stem=True, bigrams=True)
    if debug_mode:
        timings['tokenization'] = time() - start_time

    # BM25 searches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        start_time = time()
        future_body = executor.submit(BM25_search, index_body, query_tokens, N)
        future_title = executor.submit(BM25_search, index_title, query_tokens, N)
        
        bm25_body_cands = future_body.result()
        timings['bm25_body_search'] = time() - start_time

        start_time = time()
        bm25_title_cands = future_title.result()
        timings['bm25_title_search'] = time() - start_time

    # Processing and merging results
    df_body = pd.DataFrame(bm25_body_cands, columns=['doc_id', 'bm25_body_score'])
    df_title = pd.DataFrame(bm25_title_cands, columns=['doc_id', 'bm25_title_score'])
    df_scores = pd.merge(df_body, df_title, on='doc_id', how='outer').fillna(0)

    # Bigram score retrieval
    start_time = time()
    ngram_cands = find_relevant_documents(query_tokens_bigrams, index_title_bigram)[1]
    filtered_ngram_cands = {term: doc_dict for term, doc_dict in ngram_cands.items() if any(doc_id in df_scores['doc_id'].values for doc_id in doc_dict)}
    df_scores['score_ngram'] = df_scores.apply(compute_unique_term_scores, axis=1, args=(filtered_ngram_cands,))
    if debug_mode:
        timings['ngram_search'] = time() - start_time

    # Page rank and page view score retrieval
    start_time = time()
    df_scores['page_rank_score'] = df_scores.apply(get_page_rank_score, axis=1)
    df_scores['page_views_score'] = df_scores.apply(get_page_views_score, axis=1)
    if debug_mode:
        timings['page_rank_views'] = time() - start_time

    # Normalization and final scoring
    start_time = time()
    score_columns = ['bm25_body_score', 'bm25_title_score', 'score_ngram', 'page_rank_score', 'page_views_score']
    normalized_columns = [col + '_normalized' for col in score_columns]
    scaler = MinMaxScaler()
    df_scores[normalized_columns] = scaler.fit_transform(df_scores[score_columns])
    df_scores['final_score'] = df_scores[normalized_columns].mul(weights).sum(axis=1)
    top_n_documents = df_scores.sort_values(by='final_score', ascending=False).head(N)
    if debug_mode:
        timings['scoring_normalization'] = time() - start_time

    # Extracting top N document IDs and titles
    top_n_doc_ids = top_n_documents['doc_id'].values
    res = [(str(doc_id), id2title[doc_id]) for doc_id in top_n_doc_ids]

    if debug_mode:
        return res, timings
    return res

def search_engine(q , weights = [1, 1, 1, 3, 3] , N = 100):
    return information_retrieval(q, title_index, title_index_bigram,body_index,weights=weights,N=N)
