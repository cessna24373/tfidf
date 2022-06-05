#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import defaultdict
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open("ac.txt","r") as f:
    data=f.readlines()

chapters=defaultdict(str)
title="header"
for line in data:
    line=line.strip()
    if line=="":
        continue
    m=re.search("\d+\/\d+\/\d+",line)
    if m:
        continue
    if line=="APPENDIX A. STANDARD BRIEFING ELEMENTS AND RESOURCES":
        break
    m=re.search("^(\d+\s+[A-Z\s\(\)\-’]+\.*)\s*(.*)",line)
    if m:
        title=m.groups(0)[0]
        line=m.groups(0)[1]
    chapters[title]=chapters[title]+line

lemmatizer = WordNetLemmatizer()
def lemma_me(sent):
    sentence_tokens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sentence_tokens)
    sentence_lemmas = []
    for token, pos_tag in zip(sentence_tokens, pos_tags):
        if "ads-b" in token:
            sentence_lemmas.append(token)
            continue
        if pos_tag[1][0].lower() in ['n', 'v', 'a', 'r']:
            lemma = lemmatizer.lemmatize(token, pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)
    return sentence_lemmas

question="what is information"

tv = TfidfVectorizer(tokenizer=lemma_me)
tf = tv.fit_transform(list(chapters.values()))
qa=tv.transform([question])
values = cosine_similarity(qa, tf)
ind=values.argsort().flatten()[-1]

qa=tv.transform([question])
values = cosine_similarity(qa, tf)
ind=values.argsort().flatten()[-1]
lemma_me(question)

names=tv.get_feature_names()
index=names.index("information")
tf[:,index].toarray()

tv.get_feature_names()



