"""
1) To Lower
2) Remove Punct and Numbers (Except Some Specials=excluded_chars)
3) Remove Stop Words
4) Remove Duplicate Spaces
"""

import pandas as pd
from stop_words import get_stop_words

def return_preprocessed_df(to_lowercase = True, remove_punc = True, remove_stop_words = True):
    
    df = pd.read_csv("../data/corpora_dataset.csv",sep=";", encoding='utf8')

    excluded_chars = ["ù","è","à","ì","é","ò","á","ó","ü","ö","ñ","ä","í","å","â"]

    def to_lower(row):

        return row['titles'].lower()
    
    if to_lowercase:
    
        df['titles'] = df.apply(to_lower, axis = 1)

    titles = df['titles'].to_list()

    punc_dict = {}

    char_dict = {}

    for title in titles:

        for char in title:

            if ((ord(char) < 97) or (ord(char) > 122)) and (char not in excluded_chars):

                if char in punc_dict:

                    a = punc_dict[char]
                    a += 1
                    punc_dict[char] = a
                else:
                    punc_dict[char] = 1
            else:
                if char in char_dict:

                    a = char_dict[char]
                    a += 1
                    char_dict[char] = a
                else:
                    char_dict[char] = 1

    del punc_dict[" "]


    stop = get_stop_words('italian')

    for i in range(len(titles)):

        for punc in punc_dict.keys():
            
            if remove_punc:

                titles[i] = titles[i].replace(punc, "")
                
            if remove_stop_words:

                titles[i]  = ' '.join([i for i in titles[i].split() if i not in stop])

    df["titles"] = titles

    return df