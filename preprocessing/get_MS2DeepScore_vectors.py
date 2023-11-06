from ms2deepscore import MS2DeepScore
def get_MS2DeepScore_vectors(spectra,model):
    model = MS2DeepScore(model)
    vectors = model.calculate_vectors(spectra)
    return vectors