# -*- coding: utf-8 -*-
import re
from flashtext import KeywordProcessor

class KeywordCorpus(dict):
    
    def __setitem__(self, keyword, corpus):
        super().__setitem__(keyword, corpus)

    def __getitem__(self, keyword):
        return super().get(keyword, f'Corpus of Keyword {keyword} does not exist.')

class KeywordCorpusFactory(KeywordProcessor):

    def __init__(self, case_sensitive=False):
        super().__init__(case_sensitive=case_sensitive)
        self.keyword_corpus = KeywordCorpus()
        # self._keyword_corpus = KeywordCorpus()

    def create_keyword_corpus(self, keyword, sentences):

        self.add_keyword(keyword, '')

        for s in sentences:
            print(self.replace_keywords(s))

        self.keyword_corpus[keyword] = \
            list(self.replace_keywords(sentence).split(' ') for sentence in sentences)
        
    # @property
    # def keyword_corpus(self, keyword):
    #     return self._keyword_corpus[keyword]
    
    # @keyword_corpus.setter
    # def keyword_corpus(self, keyword, corpus):

    #     if keyword in self._keyword_corpus:
    #         self._keyword_corpus[keyword].extend(corpus)
    #     else:
    #         self._keyword_corpus[keyword] = corpus
        

def clean_keyword_in_sentence(keyword, sentence):
    '''
    substitute keyword in sentence to ''
    
    :param keyword: 
    :type keyword: str
    
    :param sentence: 
    :type sentence: str
    
    '''
    return re.sub(keyword, '', sentence , flags=re.I)



def write_sentence_in_dict(keyword_dict, keyword, clean_sentence):
    '''

    write preprocessed sentence in keyword_dict , ex: {cve_id : [sentence] }, if sentence duplicate, not append 
    
    :param keyword_dict: the keyword dict
    :type keyword_dict: dict

    :param keyword: 
    :type keyword: str
    
    :param clean_sentence: the preprocessed sentence   
    :type clean_sentence: str
    
    '''
    
    clean_sentence = clean_keyword_in_sentence(keyword, clean_sentence)
    #add all cve corpus to dict
    if keyword not in keyword_dict:
        keyword_dict[keyword] = [clean_sentence]
    else:
        #detect duplicate data
        if clean_sentence not in keyword_dict[keyword]:
            keyword_dict[keyword].append(clean_sentence)
