def export_benchmarking(name,preprocessing,comparison,visualization,vectorization=None):
    name = name+".txt"
    with open(name, "w") as file:
        file.write("Preprocessing: " + str(preprocessing) + "\n")
        file.write("Vectorization: " + str(vectorization) + "\n")
        file.write("Comparison: " + str(comparison) + "\n")
        file.write("Visualization: " + str(visualization) + "\n")

