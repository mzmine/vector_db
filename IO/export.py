import networkx as nx
import matchmsextras.networking as net
from matchms.networking import SimilarityNetwork


def generateNetwork(scores, spectrums):
    ms_network = SimilarityNetwork(identifier_key="spectrum_id", score_cutoff=0.7, max_links=10)
    ms_network.create_network(scores)
    our_network = ms_network.graph
    subnetworks = [our_network.subgraph(c).copy() for c in nx.connected_components(our_network)]
    nx.write_graphml(our_network, "network_GNPS_cutoff_07.graphml")
    metadata = net.extract_networking_metadata(spectrums)
    metadata.head()
    metadata.to_csv("network_GNPS_metadata.csv")