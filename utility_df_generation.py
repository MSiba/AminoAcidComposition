import pandas as pd
from distance_frequency import DistanceFrequency
from amino_acid_composition import AminoAcidSequence
from structural_protein_decomposition import ProteinStructure

__author__ = "Siba Mohsen"


class UtilsDataFrames:

    @staticmethod
    def generate_end_dataframe(list_seq, list_structure, list_label):
        """
        generates the end data frame containing the following columns:
        - peptide sequence
        - peptide structure file name
        - label
        - Encoding 1:
                *distance frequency
        - Encoding 2:
                *amino acid composition
        - Encoding 3:
                * Function 1: average distance
                * Function 2: total distance
                * Function 3: number of instances in the structure
                * Function 4: frequency of instances in the structure
                * Function 5: cartesian product
        :return: a data frame containing 10 columns.
        """
        end_data_frame = pd.DataFrame({'peptide': list_seq,
                                       'structure': list_structure,
                                       'label': list_label})

        end_data_frame['distance_frequency'] = end_data_frame['peptide'].apply(lambda x:
                                                                               DistanceFrequency(x).feature_vector())
        end_data_frame['amino_acid_composition'] = end_data_frame['peptide'].apply(lambda x:
                                                                                   AminoAcidSequence(x).composition())
        end_data_frame['f1_average_distance'] = end_data_frame['structure'].apply(lambda x:
                                                                                  ProteinStructure(x, 15)
                                                                                  .average_distance())
        end_data_frame['f2_total_distance'] = end_data_frame['structure'].apply(lambda x:
                                                                                ProteinStructure(x, 15).
                                                                                total_distance())
        end_data_frame['f3_number_instances'] = end_data_frame['structure'].apply(lambda x:
                                                                                  ProteinStructure(x, 15)
                                                                                  .number_instances())
        end_data_frame['f4_frequency_instances'] = end_data_frame['structure'].apply(lambda x:
                                                                                     ProteinStructure(x, 15)
                                                                                     .frequency_instances())
        end_data_frame['f5_cartesian_product'] = end_data_frame['structure'].apply(lambda x:
                                                                                   ProteinStructure(x, 15)
                                                                                   .cartesian_product())

        return end_data_frame

    @staticmethod
    def n_terminal_df(df, sequence_col_name, label_col_name, nb_of_residues):
        """
        create a data set without duplicates given the desired number of residues of the peptides from N-terminal.
        :param df: Dataframe to modify
        :param nb_of_residues: e.g. 5, 10, 15 to cut the N-Terminal from the peptide rest.
        :param sequence_col_name: the column, in which the slice operation should be done
        :param label_col_name: the name of the label column as the user has defined
        :return: dataset of peptide sequences of length <nb_of_residues> and their labels, without duplicates
        """
        dataset = df[[sequence_col_name, label_col_name]].copy()
        dataset[sequence_col_name] = dataset[sequence_col_name].str.slice(0, nb_of_residues, 1)
        dataset.drop_duplicates(keep='first', inplace=True)
        label_1 = dataset[dataset[label_col_name] == 1].count()
        label_0 = dataset[dataset[label_col_name] == 0].count()
        if label_1[0] > label_0[0]:
            label_sub = label_1 - label_0
            indexes_n = dataset.index[dataset['label'] == 1].tolist()
            last_n = indexes_n[-label_sub[0]:len(indexes_n)]
            dataset.drop(last_n, inplace=True)
        else:
            label_sub = label_0 - label_1
            indexes_n = dataset.index[dataset['label'] == 0].tolist()
            last_n = indexes_n[-label_sub[0]:len(indexes_n)]
            dataset.drop(last_n, inplace=True)
        return dataset

    @staticmethod
    def c_terminal_df(df, sequence_col_name, label_col_name, nb_of_residues):
        """
        create a data set without duplicates given the desired number of residues of the peptides from C-terminal.
        :param df: Dataframe from which other dataframes will be generated
        :param nb_of_residues: e.g. 5, 10, 15 to cut the C-Terminal from the peptide rest.
        :param sequence_col_name: the column, in which the slice operation should be done
        :param label_col_name: the name of the label column as the user has defined
        :return: dataset of peptide sequences of length nb_of_residues and their labels, without duplicates
        """
        dataset = df[[sequence_col_name, label_col_name]].copy()
        dataset[sequence_col_name] = dataset[sequence_col_name].str.slice(- nb_of_residues,
                                                                          len(dataset[sequence_col_name]),
                                                                          1)
        dataset.drop_duplicates(keep='first', inplace=True)
        label_1 = dataset[dataset[label_col_name] == 1].count()
        label_0 = dataset[dataset[label_col_name] == 0].count()
        if label_1[0] > label_0[0]:
            label_sub = label_1 - label_0
            indexes_n = dataset.index[dataset['label'] == 1].tolist()
            last_n = indexes_n[-label_sub[0]:len(indexes_n)]
            dataset.drop(last_n, inplace=True)
        else:
            label_sub = label_0 - label_1
            indexes_n = dataset.index[dataset['label'] == 0].tolist()
            last_n = indexes_n[-label_sub[0]:len(indexes_n)]
            dataset.drop(last_n, inplace=True)
        return dataset

    @staticmethod
    def nc_terminal_df(df, sequence_col_name, label_col_name, nb_of_residues=15):
        """
        creates a data set without duplicates given the desired number of residues to cut from the N- and C-terminal.
        :param df: Data frame from which dataframes will be generated
        :param nb_of_residues: e.g. 15 to cut the N- and C-Terminal from the peptide rest.
        :param sequence_col_name: the column, in which the slice operation should be done
        :param label_col_name: the name of the label column as the user has defined
        :return: dataset of peptide sequences of length nb_of_residues and their labels, without duplicates
        """

        ntct15 = df[[sequence_col_name, label_col_name]].copy()
        ntct15_nt15 = df[[sequence_col_name, label_col_name]].copy()
        ntct15_ct15 = df[[sequence_col_name, label_col_name]].copy()

        if len(ntct15['peptide'].map(str).apply(len)) < nb_of_residues*2:
            ntct15.drop(ntct15[ntct15['peptide'].map(str).apply(len) < nb_of_residues*2].index, inplace=True)
        else:
            ntct15_nt15['peptide'] = ntct15_nt15['peptide'].str.slice(0, nb_of_residues, 1)
            ntct15_ct15['peptide'] = ntct15_ct15['peptide'].str.slice(- nb_of_residues, len(ntct15_ct15['peptide']), 1)
            ntct15['peptide'] = ntct15_nt15['peptide'] + ntct15_ct15['peptide']

        ntct15.drop_duplicates(keep='first', inplace=True)
        label_1 = ntct15[ntct15[label_col_name] == 1].count()
        label_0 = ntct15[ntct15[label_col_name] == 0].count()
        if label_1[0] > label_0[0]:
            label_sub = label_1 - label_0
            indexes_n = ntct15.index[ntct15['label'] == 1].tolist()
            last_n = indexes_n[-label_sub[0]:len(indexes_n)]
            ntct15.drop(last_n, inplace=True)
        else:
            label_sub = label_0 - label_1
            indexes_n = ntct15.index[ntct15['label'] == 0].tolist()
            last_n = indexes_n[-label_sub[0]:len(indexes_n)]
            ntct15.drop(last_n, inplace=True)
        return ntct15

