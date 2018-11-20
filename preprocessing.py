# -*- coding: utf-8 -*-
import re

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
