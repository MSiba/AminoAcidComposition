import numpy as np
from Bio.Data import IUPACData

__author__ = "Siba Mohsen"

class Score:

    """computes the score of a data frame using the quantitative matrix"""

    def __init__(self, df, sequence_col_name, label_col_name):
        self.df = df
        self.sequence_col_name = sequence_col_name
        self.label_col_name = label_col_name

    aa_20 = tuple(IUPACData.protein_letters_1to3.keys())

    def __generate_binary_matrices(self, label):
        """
        generates matrices for each peptide of a data frame. The matrices have n rows indicating the position of each
        amino acid in the peptide, where n is in {5, 10, 15, main_data_frame_peptide_length}.
        They also enclose 20 columns, each column represents one amino acid.
        :param label: 0 or 1 to split the data into antibacterial and non-antibacterial.
        :return: a list of matrices/arrays fulfilling the guideline mentioned above.
        """
        total_matrices = []
        end = []
        result = []
        for ind in self.df[self.df[self.label_col_name] == label].index.values:
            for f in range(len(self.df.loc[ind][self.sequence_col_name])):
                for aa in self.aa_20:
                    if aa in self.df.loc[ind][self.sequence_col_name][f]:
                        result.append(1)
                    else:
                        result.append(0)
                end.append(result)
                result = []
            total_matrices.append(end)
            end = []
        return total_matrices

    def __quantitative_matrix(self):
        """
        As mentioned in Lata et al., equation(1) must be used to generate the quantitative matrix of a data frame.
        First, the data frame is divided and stored in two variables (antimicrobial and non antimicrobial) based
        on their labels (1 and 0, respectively) as matrices.
        Second, these matrices are added each together into one matrix. In order to fulfill equation(2) and equation(3)
        of the publication, the added matrices must be divided by the number of sequences existing in the data frame
        to get the probability.
        Third, the subtraction of the probability matrices introduced in eq(2) and eq(3) will be done and thereafter,
        the quantitative matrix can be seen.
        NOTE: This function takes some seconds to be calculated.
        :return: one quantitative matrix
        """
        antimicrobial = self.__generate_binary_matrices(1)
        non_antimicrobial = self.__generate_binary_matrices(0)
        s1 = sum(np.array(d) for d in antimicrobial)
        s0 = sum(np.array(d) for d in non_antimicrobial)
        probability_s1 = s1 / len(antimicrobial)
        probability_s0 = s0 / len(non_antimicrobial)
        qm = np.subtract(probability_s1, probability_s0)
        return qm

    def calculate_score(self, peptide):
        """
        given a peptide sequence, this function calculates its score by adding the scores of each residue (amino acid)
        at a specific position (rows, columns) using the quantitative matrix of the data frame. If the score > 0,
        then this means that the probability in eq(2) is greater than that of eq(3). In this case, the peptide should
        be estimated to be antimicrobial. If score < 0, the peptide is more probably to be non anti-bacterial.
        :param peptide: String. A sequence of amino acids.
        :return: int. The score of the given peptide.
        """
        score = 0
        qm = self.__quantitative_matrix()
        for num in range(len(peptide)):
            score += qm[num][self.aa_20.index(peptide[num])]
        return score

    def calculate_score_specific_qm(self, peptide, qm):
        """
        given a peptide sequence and a specific quantitative matrix, this function calculates its score by adding
        the scores of each residue (amino acid) at a specific position (rows, columns) using the quantitative matrix
        of the data frame. If the score > 0, then this means that the probability in eq(2) is greater than that
        of eq(3). In this case, the peptide should be estimated to be antimicrobial. If score < 0, the peptide is
        more probably to be non anti-bacterial.
        :param peptide: String. A sequence of amino acids.
        :param qm: List<Array>. A quantitative matrix.
        :return: int. The score of the given peptide.
        """
        score = 0
        for num in range(len(peptide)):
            score += qm[num][self.aa_20.index(peptide[num])]
        return score
