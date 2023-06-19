from matplotlib import pyplot as plt

def plotScores(scores_array):
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(scores_array[:50, :50]["ModifiedCosine_score"], cmap="viridis")
    plt.colorbar(shrink=0.7)
    plt.title("Modified Cosine spectra similarities")
    plt.xlabel("Spectrum #ID")
    plt.ylabel("Spectrum #ID")
    plt.show()

def plotScoresRestrictive(scores_array,min_match):
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(scores_array[:50, :50]["score"] *
               (scores_array[:50, :50]["matches"] >= min_match), cmap="viridis")
    plt.colorbar(shrink=0.7)
    plt.title("Spectra similarities (min_match=5)")
    plt.xlabel("Spectrum #ID")
    plt.ylabel("Spectrum #ID")
    plt.show()
