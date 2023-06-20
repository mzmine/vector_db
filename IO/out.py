from matplotlib import pyplot as plt
import networkx as nx
import matchmsextras.networking as net
from matchms.networking import SimilarityNetwork
def plotScores(scores_array):
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(scores_array[:50, :50]["score"], cmap="viridis")
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

def getBestMatches(scores, spectrums, spectrum, matches):
    best_matches = scores.scores_by_query(spectrums[spectrum], sort=True)[:matches]
    print([x[1] for x in best_matches])

def generateNetwork(scores,spectrums):
    ms_network = SimilarityNetwork(identifier_key="spectrum_id", score_cutoff=0.7, max_links=10)
    ms_network.create_network(scores)
    our_network = ms_network.graph
    subnetworks = [our_network.subgraph(c).copy() for c in nx.connected_components(our_network)]
    nx.write_graphml(our_network, "network_GNPS_cutoff_07.graphml")
    metadata = net.extract_networking_metadata(spectrums)
    metadata.head()
    metadata.to_csv("network_GNPS_metadata.csv")