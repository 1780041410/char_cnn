import re
import pandas as pd
from fuzzywuzzy import fuzz
import distance
import jieba.posseg as pseg
import jieba
SAFE_DIV = 0.0001
import math
def get_token_features(q1, q2):
    stop_file=open('data/stopwords.txt','r',encoding='utf-8')
    STOP_WORDS=stop_file.read().split('\n')
    token_features = [0.0]*10
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    def num_repeat_word(q1,q2):
        num=0
        for q in q1:
            if q in q2:
                num+=1
        return num
    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    token_features[0] =  (min(len(q1_words), len(q2_words)) + SAFE_DIV)/common_word_count
    token_features[1] =  (max(len(q1_words), len(q2_words)) + SAFE_DIV)/common_word_count
    token_features[2] =  (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)/common_stop_count
    token_features[3] =  (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)/common_stop_count
    token_features[4] =  (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)/common_token_count
    token_features[5] =  (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)/common_token_count
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features
def get_repeat_char_ratio(q1, q2):
    i=0
    for x in q1:
        if x in q2:
            i+=1
    return i/((len(q1)+len(q2))/2)
def get_pos_features(q1_pos, q2_pos):
    pos_features = [0.0] * 9
    q1_pos_tokens = q1_pos.split()
    q2_pos_tokens = q2_pos.split()
    str1=''.join(q1_pos_tokens)
    str2 = ''.join(q2_pos_tokens)
    q1_poss = set([word for word in q1_pos_tokens ])
    q2_poss = set([word for word in q2_pos_tokens ])
    common_word_count = len(q1_poss.intersection(q1_poss))
    common_token_count = len(set(q1_pos_tokens).intersection(set(q1_pos_tokens)))
    pos_features[0] = common_word_count / (min(len(q1_poss), len(q2_poss)) + SAFE_DIV)
    pos_features[1] = common_word_count / (max(len(q1_poss), len(q2_poss)) + SAFE_DIV)
    pos_features[2] = common_token_count / (min(len(q1_pos_tokens), len(q2_pos_tokens)) + SAFE_DIV)
    pos_features[3] = common_token_count / (max(len(q1_pos_tokens), len(q2_pos_tokens)) + SAFE_DIV)
    pos_features[4] = fuzz.token_set_ratio(str1,str2)
    pos_features[5] = fuzz.QRatio(str1,str2)
    pos_features[6] =fuzz.partial_ratio(str1,str2)
    pos_features[7] = distance.levenshtein(str1,str2)/((len(str1)+len(str2))/2)
    pos_features[8] = get_repeat_char_ratio(str1, str2)
    return pos_features
def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)
def get_Levenshtein(q1,q2):
    dis=distance.levenshtein(q1,q2)/((len(q1)+len(q2))/2)
    return dis
def extract_features(df):
    df["question1"] = df["question1"].fillna("")
    df["question2"] = df["question2"].fillna("")
    print("token features...")
    token_features = df.apply(lambda x: get_token_features(x["question1_tokens"], x["question2_tokens"]), axis=1)
    token_features=list(token_features.values)
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))
    pos_features = df.apply(lambda x: get_pos_features(x["question1_pos"], x["question2_pos"]), axis=1)
    df["pos_cwc_min"] = list(map(lambda x: x[0], pos_features))
    df["pos_cwc_max"] = list(map(lambda x: x[1], pos_features))
    df["pos_ctc_min"] = list(map(lambda x: x[2], pos_features))
    df["pos_ctc_max"] = list(map(lambda x: x[3], pos_features))
    df["pos_token_set_ratio"] = list(map(lambda x: x[4], pos_features))
    df["pos_QRatio"] = list(map(lambda x: x[5], pos_features))
    df["pos_partial_ratio"] = list(map(lambda x: x[6], pos_features))
    df["pos_Levenshteinr_ratio"] = list(map(lambda x: x[7], pos_features))
    df["pos_repeat_char_ratio"] = list(map(lambda x: x[8], pos_features))
    print("fuzzy features..")
    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    df["Levenshteinr_ratio"] = df.apply(lambda x: get_Levenshtein(x["question1"], x["question2"]), axis=1)
    df["repeat_char_ratio"] = df.apply(lambda x: get_repeat_char_ratio(x["question1"], x["question2"]), axis=1)
    return df
print("Extracting features for train:")
train_df = pd.read_csv("data/train.csv")
train_df = extract_features(train_df)
train_df.drop(["id", "qid1", "qid2", "question1", "question2", "label","question1_tokens","question2_tokens","question1_pos","question2_pos"], axis=1, inplace=True)
train_df.to_csv("data/nlp_features_train.csv", encoding='utf-8',index=False)
print("Extracting features for test:")
test_df = pd.read_csv("data/dev.csv")
test_df = extract_features(test_df)
test_df.drop(["id", "question1", "question2",'qid1','qid2','question1_tokens','question2_tokens','question1_pos','question2_pos'], axis=1, inplace=True)
test_df.to_csv("data/nlp_features_test.csv", encoding='utf-8',index=False)