from typing import List
import numpy as np

def bleu_score(predicted: List[int], target: List[int], N: int) -> float:
    """
    Finds the BLEU-N score between the predicted sentence and a single reference (target) sentence.
    Feel free to deviate from this skeleton if you prefer.

    Edge case: If the length of the predicted sentence or the target is less than N, return 0.
    """
    if len(predicted) < N or len(target) < N:
        # TODO
        pass
    
    def C(y, g):
        # TODO how many times does n-gram g appear in y?
        pass

    geo_mean = 1
    for n in range(1, N+1):
        grams = set() # unique n-grams
        for i in range(len(predicted)-n+1):
            # TODO add to grams
            pass
        
        numerator = None # TODO numerator of clipped precision
        denominator = None # TODO denominator of clipped precision

        geo_mean *= (numerator/denominator)**(1/N)
    
    brevity_penalty = None # TODO
    return brevity_penalty * geo_mean


