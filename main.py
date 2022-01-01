from utility_df_generation import UtilsDataFrames
from machine_learning_models import Classification
from quantitative_matrix import Score
from sklearn import preprocessing
from statistics import mean
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt


__author__ = "Siba Mohsen"


# Read the .fasta file containing 292 AMP sequences
amp_sequences = open('data/data_amps.fasta', 'r')
# store the sequences in a list
total_sequences = []
for i, line in enumerate(amp_sequences):
    if i % 2 != 0:
        total_sequences.append(line.strip())
print(total_sequences)
# to make sure if there are 292 sequences
print("The length of the AMP sequence is {}".format(len(total_sequences)))

# examine the lengths of our data set
peptides_len = []
for peptide in total_sequences:
    peptides_len.append(len(peptide))
print(peptides_len)
print("The minimum AMP length is {}.\nThe maximum AMP length is {}".format(min(peptides_len), max(peptides_len)))
print("The lengths of all peptides: {}".format(sorted(set(peptides_len))))
print("The mean peptide length of the sequences: ", mean(peptides_len))

# calculates the incidence of each length in the data
len_incidence = {}
for length in peptides_len:
    len_incidence[length] = peptides_len.count(length)
print("Incidence of each peptide length in our data: ", len_incidence)
print("The most incident peptide length: ", max(len_incidence))
print("The least incident peptide length: ", min(len_incidence))

# plot the lengths (x axis) and their incidence (y-axis)
plt.bar(list(len_incidence.keys()), len_incidence.values(), color='b')
plt.xlabel('Längen der Peptiden')
plt.ylabel('Anzahl von Peptide')
plt.savefig('./dataset.png')
plt.show()

# Read the structure names files:
amp_structures = ['data/ampsmohsen-Seq_1.pdb', 'data/ampsmohsen-Seq_2.pdb', 'data/ampsmohsen-Seq_3.pdb',
                  'data/ampsmohsen-Seq_4.pdb', 'data/ampsmohsen-Seq_5.pdb', 'data/ampsmohsen-Seq_6.pdb',
                  'data/ampsmohsen-Seq_7.pdb', 'data/ampsmohsen-Seq_8.pdb', 'data/ampsmohsen-Seq_9.pdb',
                  'data/ampsmohsen-Seq_10.pdb', 'data/ampsmohsen-Seq_11.pdb', 'data/ampsmohsen-Seq_12.pdb',
                  'data/ampsmohsen-Seq_14.pdb', 'data/ampsmohsen-Seq_15.pdb', 'data/ampsmohsen-Seq_16.pdb',
                  'data/ampsmohsen-Seq_17.pdb', 'data/ampsmohsen-Seq_19.pdb', 'data/ampsmohsen-Seq_20.pdb',
                  'data/ampsmohsen-Seq_21.pdb', 'data/ampsmohsen-Seq_24.pdb', 'data/ampsmohsen-Seq_25.pdb',
                  'data/ampsmohsen-Seq_26.pdb', 'data/ampsmohsen-Seq_27.pdb', 'data/ampsmohsen-Seq_28.pdb',
                  'data/ampsmohsen-Seq_29.pdb', 'data/ampsmohsen-Seq_30.pdb', 'data/ampsmohsen-Seq_31.pdb',
                  'data/ampsmohsen-Seq_33.pdb', 'data/ampsmohsen-Seq_34.pdb', 'data/ampsmohsen-Seq_35.pdb',
                  'data/ampsmohsen-Seq_36.pdb', 'data/ampsmohsen-Seq_38.pdb', 'data/ampsmohsen-Seq_40.pdb',
                  'data/ampsmohsen-Seq_41.pdb', 'data/ampsmohsen-Seq_42.pdb', 'data/ampsmohsen-Seq_43.pdb',
                  'data/ampsmohsen-Seq_44.pdb', 'data/ampsmohsen-Seq_45.pdb', 'data/ampsmohsen-Seq_46.pdb',
                  'data/ampsmohsen-Seq_47.pdb', 'data/ampsmohsen-Seq_50.pdb', 'data/ampsmohsen-Seq_52.pdb',
                  'data/ampsmohsen-Seq_53.pdb', 'data/ampsmohsen-Seq_55.pdb', 'data/ampsmohsen-Seq_56.pdb',
                  'data/ampsmohsen-Seq_57.pdb', 'data/ampsmohsen-Seq_58.pdb', 'data/ampsmohsen-Seq_59.pdb',
                  'data/ampsmohsen-Seq_60.pdb', 'data/ampsmohsen-Seq_62.pdb', 'data/ampsmohsen-Seq_63.pdb',
                  'data/ampsmohsen-Seq_65.pdb', 'data/ampsmohsen-Seq_68.pdb', 'data/ampsmohsen-Seq_71.pdb',
                  'data/ampsmohsen-Seq_72.pdb', 'data/ampsmohsen-Seq_74.pdb', 'data/ampsmohsen-Seq_77.pdb',
                  'data/ampsmohsen-Seq_78.pdb', 'data/ampsmohsen-Seq_80.pdb', 'data/ampsmohsen-Seq_81.pdb',
                  'data/ampsmohsen-Seq_82.pdb', 'data/ampsmohsen-Seq_83.pdb', 'data/ampsmohsen-Seq_84.pdb',
                  'data/ampsmohsen-Seq_85.pdb', 'data/ampsmohsen-Seq_86.pdb', 'data/ampsmohsen-Seq_87.pdb',
                  'data/ampsmohsen-Seq_88.pdb', 'data/ampsmohsen-Seq_89.pdb', 'data/ampsmohsen-Seq_91.pdb',
                  'data/ampsmohsen-Seq_92.pdb', 'data/ampsmohsen-Seq_94.pdb', 'data/ampsmohsen-Seq_95.pdb',
                  'data/ampsmohsen-Seq_99.pdb', 'data/ampsmohsen-Seq_100.pdb', 'data/ampsmohsen-Seq_101.pdb',
                  'data/ampsmohsen-Seq_102.pdb', 'data/ampsmohsen-Seq_103.pdb', 'data/ampsmohsen-Seq_104.pdb',
                  'data/ampsmohsen-Seq_105.pdb', 'data/ampsmohsen-Seq_107.pdb', 'data/ampsmohsen-Seq_108.pdb',
                  'data/ampsmohsen-Seq_110.pdb', 'data/ampsmohsen-Seq_111.pdb', 'data/ampsmohsen-Seq_112.pdb',
                  'data/ampsmohsen-Seq_113.pdb', 'data/ampsmohsen-Seq_115.pdb', 'data/ampsmohsen-Seq_116.pdb',
                  'data/ampsmohsen-Seq_117.pdb', 'data/ampsmohsen-Seq_119.pdb', 'data/ampsmohsen-Seq_120.pdb',
                  'data/ampsmohsen-Seq_121.pdb', 'data/ampsmohsen-Seq_122.pdb', 'data/ampsmohsen-Seq_123.pdb',
                  'data/ampsmohsen-Seq_124.pdb', 'data/ampsmohsen-Seq_125.pdb', 'data/ampsmohsen-Seq_126.pdb',
                  'data/ampsmohsen-Seq_129.pdb', 'data/ampsmohsen-Seq_131.pdb', 'data/ampsmohsen-Seq_132.pdb',
                  'data/ampsmohsen-Seq_134.pdb', 'data/ampsmohsen-Seq_137.pdb', 'data/ampsmohsen-Seq_138.pdb',
                  'data/ampsmohsen-Seq_139.pdb', 'data/ampsmohsen-Seq_141.pdb', 'data/ampsmohsen-Seq_142.pdb',
                  'data/ampsmohsen-Seq_143.pdb', 'data/ampsmohsen-Seq_144.pdb', 'data/ampsmohsen-Seq_145.pdb',
                  'data/ampsmohsen-Seq_146.pdb', 'data/ampsmohsen-Seq_147.pdb', 'data/ampsmohsen-Seq_148.pdb',
                  'data/ampsmohsen-Seq_149.pdb', 'data/ampsmohsen-Seq_150.pdb', 'data/ampsmohsen-Seq_151.pdb',
                  'data/ampsmohsen-Seq_153.pdb', 'data/ampsmohsen-Seq_154.pdb', 'data/ampsmohsen-Seq_155.pdb',
                  'data/ampsmohsen-Seq_156.pdb', 'data/ampsmohsen-Seq_159.pdb', 'data/ampsmohsen-Seq_161.pdb',
                  'data/ampsmohsen-Seq_162.pdb', 'data/ampsmohsen-Seq_164.pdb', 'data/ampsmohsen-Seq_166.pdb',
                  'data/ampsmohsen-Seq_169.pdb', 'data/ampsmohsen-Seq_171.pdb', 'data/ampsmohsen-Seq_172.pdb',
                  'data/ampsmohsen-Seq_174.pdb', 'data/ampsmohsen-Seq_175.pdb', 'data/ampsmohsen-Seq_176.pdb',
                  'data/ampsmohsen-Seq_180.pdb', 'data/ampsmohsen-Seq_182.pdb', 'data/ampsmohsen-Seq_184.pdb',
                  'data/ampsmohsen-Seq_185.pdb', 'data/ampsmohsen-Seq_186.pdb', 'data/ampsmohsen-Seq_187.pdb',
                  'data/ampsmohsen-Seq_188.pdb', 'data/ampsmohsen-Seq_189.pdb', 'data/ampsmohsen-Seq_190.pdb',
                  'data/ampsmohsen-Seq_191.pdb', 'data/ampsmohsen-Seq_196.pdb', 'data/ampsmohsen-Seq_197.pdb',
                  'data/ampsmohsen-Seq_198.pdb', 'data/ampsmohsen-Seq_199.pdb', 'data/ampsmohsen-Seq_202.pdb',
                  'data/ampsmohsen-Seq_203.pdb', 'data/ampsmohsen-Seq_204.pdb', 'data/ampsmohsen-Seq_205.pdb',
                  'data/ampsmohsen-Seq_206.pdb', 'data/ampsmohsen-Seq_207.pdb', 'data/ampsmohsen-Seq_209.pdb',
                  'data/ampsmohsen-Seq_211.pdb', 'data/ampsmohsen-Seq_212.pdb', 'data/ampsmohsen-Seq_213.pdb',
                  'data/ampsmohsen-Seq_214.pdb', 'data/ampsmohsen-Seq_215.pdb', 'data/ampsmohsen-Seq_216.pdb',
                  'data/ampsmohsen-Seq_217.pdb', 'data/ampsmohsen-Seq_219.pdb', 'data/ampsmohsen-Seq_220.pdb',
                  'data/ampsmohsen-Seq_222.pdb', 'data/ampsmohsen-Seq_223.pdb', 'data/ampsmohsen-Seq_225.pdb',
                  'data/ampsmohsen-Seq_226.pdb', 'data/ampsmohsen-Seq_227.pdb', 'data/ampsmohsen-Seq_229.pdb',
                  'data/ampsmohsen-Seq_230.pdb', 'data/ampsmohsen-Seq_231.pdb', 'data/ampsmohsen-Seq_233.pdb',
                  'data/ampsmohsen-Seq_234.pdb', 'data/ampsmohsen-Seq_235.pdb', 'data/ampsmohsen-Seq_236.pdb',
                  'data/ampsmohsen-Seq_237.pdb', 'data/ampsmohsen-Seq_239.pdb', 'data/ampsmohsen-Seq_241.pdb',
                  'data/ampsmohsen-Seq_242.pdb', 'data/ampsmohsen-Seq_243.pdb', 'data/ampsmohsen-Seq_244.pdb',
                  'data/ampsmohsen-Seq_245.pdb', 'data/ampsmohsen-Seq_246.pdb', 'data/ampsmohsen-Seq_247.pdb',
                  'data/ampsmohsen-Seq_248.pdb', 'data/ampsmohsen-Seq_249.pdb', 'data/ampsmohsen-Seq_250.pdb',
                  'data/ampsmohsen-Seq_251.pdb', 'data/ampsmohsen-Seq_252.pdb', 'data/ampsmohsen-Seq_254.pdb',
                  'data/ampsmohsen-Seq_255.pdb', 'data/ampsmohsen-Seq_256.pdb', 'data/ampsmohsen-Seq_257.pdb',
                  'data/ampsmohsen-Seq_258.pdb', 'data/ampsmohsen-Seq_259.pdb', 'data/ampsmohsen-Seq_260.pdb',
                  'data/ampsmohsen-Seq_261.pdb', 'data/ampsmohsen-Seq_262.pdb', 'data/ampsmohsen-Seq_263.pdb',
                  'data/ampsmohsen-Seq_265.pdb', 'data/ampsmohsen-Seq_266.pdb', 'data/ampsmohsen-Seq_267.pdb',
                  'data/ampsmohsen-Seq_268.pdb', 'data/ampsmohsen-Seq_269.pdb', 'data/ampsmohsen-Seq_271.pdb',
                  'data/ampsmohsen-Seq_273.pdb', 'data/ampsmohsen-Seq_275.pdb', 'data/ampsmohsen-Seq_277.pdb',
                  'data/ampsmohsen-Seq_278.pdb', 'data/ampsmohsen-Seq_279.pdb', 'data/ampsmohsen-Seq_281.pdb',
                  'data/ampsmohsen-Seq_282.pdb', 'data/ampsmohsen-Seq_283.pdb', 'data/ampsmohsen-Seq_284.pdb',
                  'data/ampsmohsen-Seq_285.pdb', 'data/ampsmohsen-Seq_286.pdb', 'data/ampsmohsen-Seq_287.pdb',
                  'data/ampsmohsen-Seq_289.pdb', 'data/ampsmohsen-Seq_290.pdb', 'data/ampsmohsen-Seq_292.pdb',
                  'data/ampsmohsen-Seq_293.pdb', 'data/ampsmohsen-Seq_296.pdb', 'data/ampsmohsen-Seq_297.pdb',
                  'data/ampsmohsen-Seq_298.pdb', 'data/ampsmohsen-Seq_299.pdb', 'data/ampsmohsen-Seq_300.pdb',
                  'data/ampsmohsen-Seq_301.pdb', 'data/ampsmohsen-Seq_302.pdb', 'data/ampsmohsen-Seq_303.pdb',
                  'data/ampsmohsen-Seq_304.pdb', 'data/ampsmohsen-Seq_305.pdb', 'data/ampsmohsen-Seq_306.pdb',
                  'data/ampsmohsen-Seq_308.pdb', 'data/ampsmohsen-Seq_309.pdb', 'data/ampsmohsen-Seq_311.pdb',
                  'data/ampsmohsen-Seq_312.pdb', 'data/ampsmohsen-Seq_313.pdb', 'data/ampsmohsen-Seq_314.pdb',
                  'data/ampsmohsen-Seq_315.pdb', 'data/ampsmohsen-Seq_317.pdb', 'data/ampsmohsen-Seq_318.pdb',
                  'data/ampsmohsen-Seq_319.pdb', 'data/ampsmohsen-Seq_320.pdb', 'data/ampsmohsen-Seq_321.pdb',
                  'data/ampsmohsen-Seq_322.pdb', 'data/ampsmohsen-Seq_323.pdb', 'data/ampsmohsen-Seq_324.pdb',
                  'data/ampsmohsen-Seq_325.pdb', 'data/ampsmohsen-Seq_327.pdb', 'data/ampsmohsen-Seq_328.pdb',
                  'data/ampsmohsen-Seq_329.pdb', 'data/ampsmohsen-Seq_330.pdb', 'data/ampsmohsen-Seq_334.pdb',
                  'data/ampsmohsen-Seq_335.pdb', 'data/ampsmohsen-Seq_336.pdb', 'data/ampsmohsen-Seq_337.pdb',
                  'data/ampsmohsen-Seq_338.pdb', 'data/ampsmohsen-Seq_339.pdb', 'data/ampsmohsen-Seq_340.pdb',
                  'data/ampsmohsen-Seq_342.pdb', 'data/ampsmohsen-Seq_343.pdb', 'data/ampsmohsen-Seq_345.pdb',
                  'data/ampsmohsen-Seq_346.pdb', 'data/ampsmohsen-Seq_348.pdb', 'data/ampsmohsen-Seq_349.pdb',
                  'data/ampsmohsen-Seq_350.pdb', 'data/ampsmohsen-Seq_351.pdb', 'data/ampsmohsen-Seq_353.pdb',
                  'data/ampsmohsen-Seq_354.pdb', 'data/ampsmohsen-Seq_355.pdb', 'data/ampsmohsen-Seq_356.pdb',
                  'data/ampsmohsen-Seq_358.pdb', 'data/ampsmohsen-Seq_359.pdb', 'data/ampsmohsen-Seq_361.pdb',
                  'data/ampsmohsen-Seq_363.pdb', 'data/ampsmohsen-Seq_364.pdb', 'data/ampsmohsen-Seq_368.pdb',
                  'data/ampsmohsen-Seq_369.pdb', 'data/ampsmohsen-Seq_370.pdb', 'data/ampsmohsen-Seq_373.pdb',
                  'data/ampsmohsen-Seq_374.pdb', 'data/ampsmohsen-Seq_376.pdb', 'data/ampsmohsen-Seq_377.pdb',
                  'data/ampsmohsen-Seq_378.pdb', 'data/ampsmohsen-Seq_379.pdb', 'data/ampsmohsen-Seq_380.pdb',
                  'data/ampsmohsen-Seq_381.pdb', 'data/ampsmohsen-Seq_382.pdb', 'data/ampsmohsen-Seq_383.pdb',
                  'data/ampsmohsen-Seq_385.pdb', 'data/ampsmohsen-Seq_386.pdb', 'data/ampsmohsen-Seq_387.pdb',
                  'data/ampsmohsen-Seq_388.pdb', 'data/ampsmohsen-Seq_391.pdb', 'data/ampsmohsen-Seq_392.pdb',
                  'data/ampsmohsen-Seq_394.pdb']

# Read the labels file
amp_label = open('data/data_amps_classes.txt', 'r')
seq_labels = []
for j, label in enumerate(amp_label):
    seq_labels.append(int(label.strip()))
print(seq_labels)
# to make sure if there are 292 labels
print("The length of the AMP labels sequence is {}".format(len(seq_labels)))

l0, l1 = 0, 0
for lab in seq_labels:
    if lab == 0:
        l0 += 1
    else:
        l1 += 1
print('l0 {} and l1 {}'.format(l0, l1))
# plot the lengths (x axis) and their incidence (y-axis)
plt.bar(['0', '1'], [l0, l1], width=0.4, color='b')
plt.xlabel('Labels: nicht antimikrobielle Petide (0) und antimikrobielle Peptide (1)')
plt.ylabel('Anzahl von Peptide')
plt.savefig('./labels.png')
plt.show()


# ###################################################################################################################
# ###################################################################################################################
# Generating data frames
# ###################################################################################################################
# ###################################################################################################################

# To generate the initial data frame
initial_data_frame = UtilsDataFrames.generate_end_dataframe(total_sequences, amp_structures, seq_labels)
print(initial_data_frame.head())

# To generate data frames from the initial data frame df
nt_5 = UtilsDataFrames.n_terminal_df(initial_data_frame, 'peptide', 'label', 5)
nt_10 = UtilsDataFrames.n_terminal_df(initial_data_frame, 'peptide', 'label', 10)
nt_15 = UtilsDataFrames.n_terminal_df(initial_data_frame, 'peptide', 'label', 15)
ct_5 = UtilsDataFrames.c_terminal_df(initial_data_frame, 'peptide', 'label', 5)
ct_10 = UtilsDataFrames.c_terminal_df(initial_data_frame, 'peptide', 'label', 10)
ct_15 = UtilsDataFrames.c_terminal_df(initial_data_frame, 'peptide', 'label', 15)
nt_ct_15 = UtilsDataFrames.nc_terminal_df(initial_data_frame, 'peptide', 'label', 15)


# ###################################################################################################################
# ###################################################################################################################
# Evaluation
# ###################################################################################################################
# ###################################################################################################################


print('------------------------------------------------------------------------------------------------------------')
print('                                         DISTANCE FREQUENCY')
print('------------------------------------------------------------------------------------------------------------')
X = preprocessing.normalize(initial_data_frame['distance_frequency'].tolist(), norm='l2')
y = np.array(initial_data_frame['label'])
print("The normalized array of feature vector", X)
Classification_feature_vector = Classification(X, y).predict()
print(Classification_feature_vector)

print('------------------------------------------------------------------------------------------------------------')
print('                                         Quantitative Matrix')
print('------------------------------------------------------------------------------------------------------------')
for generated_df in [nt_5, nt_10, nt_15, ct_5, ct_10, ct_15, nt_ct_15]:
    generated_df['score'] = generated_df['peptide'].apply(lambda x:
                                                          Score(generated_df, 'peptide', 'label').
                                                          calculate_score(x))
    print("######################## *************************************************+* ###########################")
    X_score = np.array(generated_df['score']).reshape(-1, 1)
    y_score = np.array(generated_df['label'])
    classification_quantitative_matrix = Classification(X_score, y_score).predict()
    print(classification_quantitative_matrix)


print('------------------------------------------------------------------------------------------------------------')
print('                                         Amino Acid Composition')
print('------------------------------------------------------------------------------------------------------------')
X_aacomposition = initial_data_frame['amino_acid_composition'].tolist()
Classification_amino_acid_composition = Classification(X_aacomposition, y).predict()
print(Classification_amino_acid_composition)

Classification_feature_vector = Classification(X, y).permutation_predict()
print(Classification_feature_vector)

print('------------------------------------------------------------------------------------------------------------')
print('                                       Encoding 3: Average Distances')
print('------------------------------------------------------------------------------------------------------------')
X_f1 = initial_data_frame['f1_average_distance'].tolist()
classification_average_distance = Classification(X_f1, y).predict()
print(classification_average_distance)

print('------------------------------------------------------------------------------------------------------------')
print('                                        Encoding 3: Total Distances')
print('------------------------------------------------------------------------------------------------------------')
X_f2 = initial_data_frame['f2_total_distance'].tolist()
classification_total_distance = Classification(X_f2, y).predict()
print(classification_total_distance)

print('------------------------------------------------------------------------------------------------------------')
print('                                        Encoding 3: Number of Instances')
print('------------------------------------------------------------------------------------------------------------')
X_f3 = initial_data_frame['f3_number_instances'].tolist()
classification_number_instances = Classification(X_f3, y).predict()
print(classification_number_instances)

print('------------------------------------------------------------------------------------------------------------')
print('                                        Encoding 3: Frequency of Instances')
print('------------------------------------------------------------------------------------------------------------')
X_f4 = initial_data_frame['f4_frequency_instances'].tolist()
classification_frequency_instances = Classification(X_f4, y).predict()
print(classification_frequency_instances)

print('------------------------------------------------------------------------------------------------------------')
print('                                        Encoding 3: Cartesian Product')
print('------------------------------------------------------------------------------------------------------------')
X_f5 = initial_data_frame['f5_cartesian_product'].tolist()
classification_cartesian_product = Classification(X_f5, y).predict()
print(classification_cartesian_product)

# ###################################################################################################################
# ###################################################################################################################
# Permutation tests
# ###################################################################################################################
# ###################################################################################################################

print('------------------------------------------------------------------------------------------------------------')
print('                                Permutation test: DISTANCE FREQUENCY')
print('------------------------------------------------------------------------------------------------------------')
X = preprocessing.normalize(initial_data_frame['distance_frequency'].tolist(), norm='l2')
y = np.array(initial_data_frame['label'])
print("The normalized array of feature vector", X)
permutation_feature_vector = Classification(X, y).permutation_predict()
print(permutation_feature_vector)

print('------------------------------------------------------------------------------------------------------------')
print('                                  Permutation test: Quantitative Matrix')
print('------------------------------------------------------------------------------------------------------------')
for perm_generated_df in [nt_5, nt_10, nt_15, ct_5, ct_10, ct_15, nt_ct_15]:
    print("################################ +++++++++++++++++++++++++++ ####################################")
    # permute the labels
    perm_generated_df['original_label'] = perm_generated_df['label']
    perm_generated_df['label'] = np.random.permutation(perm_generated_df['label'])
    # transform the score into a binary variable
    perm_generated_df['binary_score'] = np.where(perm_generated_df['score'] >= 0, 1, 0)
    # compare the label and the binary score
    perm_generated_df['same'] = perm_generated_df['label'] == perm_generated_df['binary_score']
    # predictions and real labels evaluation with metrics
    y_predicted_permuted_labels = np.array(perm_generated_df['binary_score'])
    y_true_permuted_labels = np.array(perm_generated_df['label'])
    print(confusion_matrix(y_true_permuted_labels, y_predicted_permuted_labels))
    accuracy_permuted_qm = accuracy_score(y_true_permuted_labels, y_predicted_permuted_labels)
    print("Accuracy: ", accuracy_permuted_qm)
    f1_score_permuted_qm = f1_score(y_true_permuted_labels, y_predicted_permuted_labels)
    print("F1 Score: ", f1_score_permuted_qm)
    mcc_permuted_qm = matthews_corrcoef(y_true_permuted_labels, y_predicted_permuted_labels)
    print("MCC: ", mcc_permuted_qm)
    roc_permuted_qm = roc_auc_score(y_true_permuted_labels, y_predicted_permuted_labels)
    print("ROC AUC: ", roc_permuted_qm)


print('------------------------------------------------------------------------------------------------------------')
print('                                Permutation test: Amino Acid Composition')
print('------------------------------------------------------------------------------------------------------------')
X_aacomposition = initial_data_frame['amino_acid_composition'].tolist()
permutation_amino_acid_composition = Classification(X_aacomposition, y).permutation_predict()
print(permutation_amino_acid_composition)

print('------------------------------------------------------------------------------------------------------------')
print('                                Permutation test: Encoding 3: Average Distances')
print('------------------------------------------------------------------------------------------------------------')
X_f1 = initial_data_frame['f1_average_distance'].tolist()
permutation_average_distance = Classification(X_f1, y).permutation_predict()
print(permutation_average_distance)

print('------------------------------------------------------------------------------------------------------------')
print('                                Permutation test: Encoding 3: Total Distances')
print('------------------------------------------------------------------------------------------------------------')
X_f2 = initial_data_frame['f2_total_distance'].tolist()
permutation_total_distance = Classification(X_f2, y).permutation_predict()
print(permutation_total_distance)

print('------------------------------------------------------------------------------------------------------------')
print('                                Permutation test: Encoding 3: Number of Instances')
print('------------------------------------------------------------------------------------------------------------')
X_f3 = initial_data_frame['f3_number_instances'].tolist()
permutation_number_instances = Classification(X_f3, y).permutation_predict()
print(permutation_number_instances)

print('------------------------------------------------------------------------------------------------------------')
print('                                Permutation test: Encoding 3: Frequency of Instances')
print('------------------------------------------------------------------------------------------------------------')
X_f4 = initial_data_frame['f4_frequency_instances'].tolist()
permutation_frequency_instances = Classification(X_f4, y).permutation_predict()
print(permutation_frequency_instances)

print('------------------------------------------------------------------------------------------------------------')
print('                                Permutation test: Encoding 3: Cartesian Product')
print('------------------------------------------------------------------------------------------------------------')
X_f5 = initial_data_frame['f5_cartesian_product'].tolist()
permutation_cartesian_product = Classification(X_f5, y).permutation_predict()
print(permutation_cartesian_product)
