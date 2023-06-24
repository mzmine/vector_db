

def getBestMatches(scores, spectrums, spectrum, matches):
    best_matches = scores.scores_by_query(spectrums[spectrum], sort=True)[:matches]
    print([x[1] for x in best_matches])

