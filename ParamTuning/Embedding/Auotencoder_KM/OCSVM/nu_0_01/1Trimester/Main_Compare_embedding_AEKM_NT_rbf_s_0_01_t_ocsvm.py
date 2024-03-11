# Main
from Utils import *
from collections import Counter
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import ParameterGrid
from sklearn.svm import OneClassSVM
import numpy as np

# WEEK DIRECTORY
#dir_week ='/Users/utente/Desktop/Varcovid/SECONDO_ANNO_ANALISI/Comparing_Different_Embedding/dataset_nov_2023_little_kmers_World/' # define the path of directory
dir_week ='/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_little_kmers_World/' # define the path of directory


# column Variant --> labels
#metadata = pd.read_csv('/mnt/resources/2022_04/2022_04/filtered_metadata_0328_weeks.csv')
metadata = pd.read_csv('/blue/salemi/share/varcovid/SECONDO_ANNO/filtered_metadatataset_Nov2023_edit_221223_World.csv') # read the metadata file (CSV)
metadata['Pango.lineage'] = metadata['Pango.lineage'].replace(' ', 'unknown') # Replace the blanks with "unknown"
id_unknown = metadata[metadata['Pango.lineage'] == 'unknown']['Accession.ID'].tolist() # Find id of "unknown"

"""

Reading files 

"""
""" When each variant has been classified as VOI or VOC"""
beta = []
measure_sensibilit=[]
NF = []  # sta per number feature
results_fine_tune = []
Index=[]
# header
# header = pd.read_csv('/mnt/resources/2022_04/2022_04/dataset_week/1/EPI_ISL_489939.csv', nrows=1)
header = pd.read_csv(dir_week+'1/EPI_ISL_529217.csv', nrows=1) # we read the CSV file
"""
Useful variable

"""
#path_salvataggio_file='/mnt/resources/2022_04/2022_04/'
path_save_file= '/blue/salemi/share/varcovid/SECONDO_ANNO/Pipeline/AutoEncoder_KM/AIME/Official_simulation/nu_0_01/1Trimester/' # where save the file
# columns in metadata
col_class_lineage = 'Pango.lineage'
col_submission_date = 'Collection.date'
col_lineage_id = 'Accession.ID'

## Processing of Data

# Define FDLs
valid_lineage_lineage = lineages_of_interest()  # mettere i lineage che definisco come classe
valid_lineage = generate_sublineages(valid_lineage_lineage) # create the sublineages
metadata[col_class_lineage] = metadata[col_class_lineage].apply(lambda x: 'unknown' if x not in valid_lineage else x) # Replacement of non-FDLs by unknown.

# Define retraining week
retraining_week = retraining_weeks() # set week of retraining use lineages
non_neutral_variants = metadata[col_class_lineage].unique().tolist() # FDLS
non_neutral_variants.remove('unknown') # remouve from the list the unknown

# kmers features
features = header.columns[1:].tolist() # kmers defined

date_format = "%Y-%m-%d"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        # logging.FileHandler('/mnt/resources/2022_04/2022_04/'+'run_main_oneclass_retrain_tmp.log', 'w+'),
                        logging.FileHandler(path_save_file + 'run_main_oneclass_retrain_tmp.log', 'w+'),
                        logging.StreamHandler()
                    ])


## Build Training set
starting_week = 1 # First week of training.

## Loading first training set
print('Built the training set')
df_trainstep_1, train_w_list = load_data(dir_week, [1]) # First training set.
train_step1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()
y_train_initial = metadata[metadata[col_lineage_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_lineage]  # Elements of training set.
lineages_train = np.array(y_train_initial.tolist())

n_training = 14
for i in range(2,n_training):
    print(i)
    df_trainstep_1, train_w_list = load_data(dir_week, [i]) # First training set.
    train_step_1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()
    y_train_initial = metadata[metadata[col_lineage_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_lineage]  # Elements of training set.
    lineages_class = np.array(y_train_initial.tolist())  # Type of lineages.
    train_step1 = np.concatenate((train_step1 ,train_step_1))  # (rw = retraining week)
    lineages_train = np.concatenate((lineages_train,lineages_class))

print('End read the file')
# Control if length is the same
print(len(train_step1))
print(len(lineages_train))

ind_scaling = 0

# Feature importance
sum_train = np.sum(train_step1, axis=0)
keepFeature=sum_train/len(train_step1)
i_no_zero = np.where(keepFeature >= 0.01)[0]
counter_i = Counter(lineages_train)  # Count the diffrent number of lineages

# filtering out features
train_step_completo = train_step1
train_step1 = train_step1[:, i_no_zero] # Filter with the importatnt features
pca_class = np.ones(len(train_step1))

#Create embeddings
#tsne = visualize_embeddings_with_tsne(train_step1,tsne_class,5, title="t-SNE Visualization of Embeddings")

train = train_step1.copy()

# Autoencoder dimension
encoding_dim = 32
autoencoder, encoder = build_deep_autoencoder(len(i_no_zero), encoding_dim)
autoencoder.fit(train, train, epochs=50, batch_size=64)
train = encoder.predict(train)

print('----------------------------------------------------------------------------------------------------------------------------')
print(train_step1.shape[1])
print('----------------------------------------------------------------------------------------------------------------------------')

# training one class
p_grid = {'kernel': ['rbf'], 'gamma': ['scale'], 'nu': [0.01],'shrinking': [True]}

all_combo = list(ParameterGrid(p_grid))

for combo in all_combo[0:1]:
    logging.info("---> One Class SVM - Param: " + str(combo))
    clf_c = OneClassSVM(**combo)
    clf_c.fit(train)
    y_test_dict_variant_type = {}
    y_test_dict_finalclass = {}
    y_test_dict_predictedclass = {}
    train_completo = train # contain the sequence with all the kmers
    for week in range(n_training, 54):  # metadata['week'].max()
        logging.info("# Week " + str(starting_week + week))

        # Loading first test step
        df_teststep_i, test_w_list = load_data(dir_week, [starting_week + week])
        test_step_i = df_teststep_i.iloc[:, 1:len(df_teststep_i.columns)].to_numpy()
        test_step_i = test_step_i[:, i_no_zero]
        test_step_i = encoder.predict(test_step_i)
        test_step_completo = test_step_i

        # Classes
        y_test_step_i = get_variant_class(metadata, df_teststep_i.iloc[:, 0].tolist())
        y_test_dict_variant_type[starting_week + week] = y_test_step_i
        y_test_fclass_i = map_variant_to_finalclass(y_test_step_i, valid_lineage)
        i_voc = np.where(np.array(y_test_fclass_i) == -1)[0]
        lineages_type = metadata[metadata[col_lineage_id].isin(df_teststep_i.iloc[:, 0].tolist())][col_class_lineage]  # Type of lineages in the week of simulation.
        lineages_test = np.array(lineages_type.tolist())
        y_test_dict_finalclass[starting_week + week] = y_test_fclass_i
        lineage_dict = Counter(y_test_step_i)

        # predict
        # Returns -1 for outliers and 1 for inliers.
        y_test_i_predict = clf_c.predict(test_step_i)

        # Store the embedding
        train_completo = np.concatenate((train_completo, test_step_completo)) # contain all sequence
        lineages_train = np.concatenate((lineages_train, lineages_test))

        # Store the classes
        y_test_dict_predictedclass[starting_week + week] = y_test_i_predict
        y_test_voc_predict = np.array(y_test_i_predict)[i_voc]

        # Create logging
        logging.info("Number of variants in week:" + str(test_step_i.shape[0]))
        logging.info("Number of lineages dominant in week:" + str(len([x for x in y_test_fclass_i if x == -1])))
        logging.info("Distribution of dominant lineages:" + str(Counter(y_test_step_i)))
        logging.info("Number of variants predicted as anomalty:" + str(
        len([x for x in y_test_dict_predictedclass[starting_week + week] if x == -1])))
        acc_voc = len([x for x in y_test_voc_predict if x == -1])
        logging.info("Number of VOC variants predicted as anomalty:" + str(acc_voc))

        for k in lineage_dict.keys():
            i_k = np.where(np.array(y_test_step_i) == k)[0]
            logging.info('Number of ' + str(k) + ' variants:' + str(len(i_k)) + '; predicted anomalty=' + str(
                len([x for x in y_test_i_predict[i_k] if x == -1])))
            h = len([x for x in y_test_i_predict[i_k] if x == -1])
            Prova = [k, h, week]
            beta.append(Prova)
            measure_variant = [k, len(i_k), h, week]  # rappresenta la variante,i casi in quella settimana,
            measure_sensibilit.append(measure_variant)

    # saving results for this comb of param of the oneclass_svm
    results = {'y_test_variant_type': y_test_dict_variant_type,
           'y_test_final_class': y_test_dict_finalclass,
           'y_test_predicted_class': y_test_dict_predictedclass}

results_fine_tune.append(results)

print('---------------------------------Vettori di introduzione alle funzioni costruite-------------------------------------')
print(beta)
print(measure_sensibilit)
print('----------------------------------------------------------------------------------------------------------------------')
# Calcolo flag

y_true_model0 = results_fine_tune[0]['y_test_final_class']
y_predict_model0 = results_fine_tune[0]['y_test_predicted_class']

fp_list = []
n_list = []
fn_list = []
n_outlier_list = []

for k in y_true_model0.keys():
    yt = np.array(y_true_model0[k])
    yp = np.array(y_predict_model0[k])

    i_inlier = np.where(yt == 1)[0]
    n_fp = len(np.where(yp[i_inlier] == -1)[0])

    fp_list.append(n_fp)
    n_list.append(len(i_inlier))

    i_outlier = np.where(yt == -1)[0]
    n_fn = len(np.where(yp[i_outlier] == 1)[0])
    fn_list.append(n_fn)
    n_outlier_list.append(len(i_outlier))


tn_list = []
tp_list = []
prec_list = []
recall_list = []
spec_list = []
f1_list = []
bal_acc = []
fpr_list = []  # List to store False Positive Rate

for i in range(len(fp_list)):
    tp = n_outlier_list[i] - fn_list[i]
    tn = n_list[i] - fp_list[i]
    tn_list.append(tn)
    tp_list.append(tp)

    if tp + fp_list[i] != 0:
        prec = tp / (tp + fp_list[i])
    else:
        prec = 0

    if tp + fn_list[i] != 0:
        rec = tp / (tp + fn_list[i])
    else:
        rec = 0

    if tn + fp_list[i] != 0:
        spec = tn / (tn + fp_list[i])
    else:
        spec = 0

    if prec + rec != 0:
        f1 = 2 * prec * rec / (prec + rec)
    else:
        f1 = 0

    if spec + rec != 0:
        balanced_accuracy = (spec + rec) / 2
    else:
        balanced_accuracy = 0

    # Calculate False Positive Rate
    if tn + fp_list[i] != 0:
        fpr = fp_list[i] / (fp_list[i] + tn)
    else:
        fpr = 0

    f1_list.append(f1)
    spec_list.append(spec)
    prec_list.append(prec)
    recall_list.append(rec)
    bal_acc.append(balanced_accuracy)
    fpr_list.append(fpr)

df_conf = pd.DataFrame({
    'TN': tn_list,
    'FP': fp_list,
    'FN': fn_list,
    'TP': tp_list,
    'Precision': prec_list,
    'Recall': recall_list,
    'F1': f1_list,
    'Specificity': spec_list,
    'Balanced_Accuracy': bal_acc,
    'FPR': fpr_list  # Add the FPR to the DataFrame
})

# df_conf.to_csv('/mnt/resources/2022_04/2022_04/conf_mat_over_time.tsv', sep='\t', index=None)
df_conf.to_csv(path_save_file + 'conf_mat_over_time_ocsvm.tsv', sep='\t', index=None)

