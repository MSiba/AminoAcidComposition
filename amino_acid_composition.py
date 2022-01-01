from utility import Utils

__author__ = "Siba Mohsen"

class AminoAcidSequence:
    """calculates the composition of a given sequence of amino acid according to the 20 residues"""

    def __init__(self, seq):
        self.seq = seq

    def composition(self):
        """
        calculates the amino acid composition in each peptide of the data frame and stores the result as a list of
        compositions(%) in a new list <column called 'amino acid composition'> using the following formula:
        composition(i) = SUM((amino acid i * 100)/length(sequence)).
        :param data_frame: pd.DataFrame. The data frame containing the peptides.
        :return: list<list<float>>. The composition of each peptide of the data frame to be given to SVM, ANN etc ...
        """
        return [((self.seq.count(aa) * 100) / len(self.seq)) if aa in self.seq else 0 for aa in Utils.aa_20]
