import json

def export_benchmarking(name,preprocessing,comparison,visualization,vectorization=None):

    data = {
        "Preprocessing": preprocessing,
        "Vectorization": vectorization,
        "Comparison": comparison,
        "Visualization": visualization
    }

    name = name + ".json"
    with open(name, "w") as file:
        json.dump(data, file, indent=4)

