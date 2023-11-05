from rdkit import Chem
from rdkit.Chem import Draw
def plotSmiles(spectra,indexes):
    for i,spectrum in enumerate(spectra):
        for j in range(indexes.shape[0]):
            if i == indexes[j]:
                m = Chem.MolFromSmiles(spectrum.get("smiles"))
                Draw.MolToFile(m, f"compound_{i}.png")
