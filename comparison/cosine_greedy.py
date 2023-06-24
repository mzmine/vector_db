from matchms import calculate_scores
from matchms.similarity import CosineGreedy

def cosine_greedy(tolerance, spectrums1, spectrums2=None):
    similarity_measure = CosineGreedy(tolerance=tolerance)
    if spectrums2 == None:
        scores = calculate_scores(spectrums1, spectrums1, similarity_measure, is_symmetric=True)
        return scores
    else:
        scores = calculate_scores(spectrums1, spectrums2, similarity_measure, is_symmetric=False)
        return scores