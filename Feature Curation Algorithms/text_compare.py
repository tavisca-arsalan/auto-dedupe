import math
from nltk.util import ngrams
from stringdist import levenshtein


def compute_jaccard_index(set_1, set_2):
    if set_1 and set_2:
        n = len(set_1.intersection(set_2))
        return n / float(len(set_1) + len(set_2) - n)
    else:
        return 0


def token_set_similarity_with_jaccard_index(str1,str2):
    try:
        if (str1 and str2) and (str1!=" " and str2!=" "):
            set1 = set(str1.split())
            set2 = set(str2.split())
            return compute_jaccard_index(set1,set2)
        else:
            raise Exception('One or more strings passed as parameters were empty or blank space!')
    except Exception as error:
        print('Exception: ' + repr(error))
        return


def get_term_frequency(term_list):
    term_frequency_map = {}
    for term in term_list:
        if term in term_frequency_map:
            term_frequency_map[term] += 1
        else:
            term_frequency_map[term] = 1
    return term_frequency_map


def cosine_similarity(text1, text2):
    a = get_term_frequency(text1.split())
    b = get_term_frequency(text2.split())
    dot_product = calculate_dot_product(a,b)
    magnitude_a = calculate_word_vector_magnitude(a)
    magnitude_b = calculate_word_vector_magnitude(b)
    return dot_product / math.sqrt(magnitude_a * magnitude_b);


def calculate_dot_product(a,b):
    set_a = set(a)
    set_b = set(b)
    intersection_set = set_a.intersection(set_b)
    dot_product = 0
    for item in intersection_set:
        dot_product += a[item] * b[item]
    return dot_product


def calculate_word_vector_magnitude(word_vector):
    magnitude=0
    for word in word_vector.keys():
        magnitude += math.pow(word_vector[word], 2)
    return magnitude


def generate_three_gram_token_set(text):
    three_gram_token_set=set()
    tokens = text.split()
    for token in tokens:
        three_grams = list(ngrams(token,3))
        three_gram_token_set.update(generate_three_gram_words(three_grams))
    return three_gram_token_set


def generate_three_gram_words(three_gram_tokens):
    three_gram_word_list =[]
    for three_gram in three_gram_tokens:
        three_gram_word_list.append(''.join(three_gram))
    return three_gram_word_list


def calculate_three_gram_name_similarity(text1,text2):
    set1 = generate_three_gram_token_set(text1)
    set2 = generate_three_gram_token_set(text2)
    score = compute_jaccard_index(set1,set2)
    return score


def calculate_levenshtein_distance(str1,str2):
    return levenshtein(str1, str2)
