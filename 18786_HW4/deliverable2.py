from typing import List
import numpy as np



def create_ngrams(x:List, n:int)->List:
    n_gram = []
    if len(x) < n:
        return None
    for i in range(len(x)-n+1):
        # print(tuple(x[i:i+n]))
        n_gram.append(tuple(x[i:i+n]))
    return n_gram

# def count_ngram(sentence, ngram):
#     # TODO how many times does n-gram g appear in y?
#     n_gram_list = create_ngrams(sentence, len(ngram))
#     return n_gram_list.count(ngram)


def bleu_score(predicted: List[int], target: List[int], N: int) -> float:
    """
    Finds the BLEU-N score between the predicted sentence and a single reference (target) sentence.
    Feel free to deviate from this skeleton if you prefer.

    Edge case: If the length of the predicted sentence or the target is less than N, return 0.
    """
    if len(predicted) < N or len(target) < N:
        return 0  # Edge case
    
    s, s_hat = len(target), len(predicted)
    geo_mean = 1
    
    for n in range(1, N+1):
        predicted_ngram = create_ngrams(predicted,n)
        original_ngram = create_ngrams(target, n)
        
        # unique n-grams in y_hat
        unique_predicted_n_gram = list(set(predicted_ngram))
        counter = 0
        for i in range(len(unique_predicted_n_gram)):
            g = unique_predicted_n_gram[i]
            C_y_hat_g = predicted_ngram.count(g)
            C_y_g = original_ngram.count(g)
            counter_g = min(C_y_hat_g, C_y_g)
            counter += counter_g 
        
        # TODO numerator of clipped precision
        numerator = counter
        # TODO denominator of clipped precision
        denominator =  s_hat- n + 1 
        geo_mean *= (numerator/denominator)**(1/N)
    # TODO
    brevity_penalty = min(1, np.exp(1- (s/s_hat)) )
    return brevity_penalty * geo_mean


# def bleu_score(predicted: List[int], target: List[int], N: int) -> float:
#     """
#     Finds the BLEU-N score between the predicted sentence and a single reference (target) sentence.
#     Feel free to deviate from this skeleton if you prefer.

#     Edge case: If the length of the predicted sentence or the target is less than N, return 0.
#     """
#     if len(predicted) < N or len(target) < N:
#         # TODO
#         pass
    
#     def C(y, g):
#         # TODO how many times does n-gram g appear in y?
#         pass

#     geo_mean = 1
#     for n in range(1, N+1):
#         grams = set() # unique n-grams
#         for i in range(len(predicted)-n+1):
#             # TODO add to grams
#             pass
        
#         numerator = None # TODO numerator of clipped precision
#         denominator = None # TODO denominator of clipped precision

#         geo_mean *= (numerator/denominator)**(1/N)
    
#     brevity_penalty = None # TODO
#     return brevity_penalty * geo_mean


