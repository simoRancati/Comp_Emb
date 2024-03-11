# Main
from Utils import *
import logging
from sklearn.svm import OneClassSVM
import numpy as np

# WEEK DIRECTORY
#dir_week ='/Users/utente/Desktop/Varcovid/SECONDO_ANNO_ANALISI/Comparing_Different_Embedding/dataset_nov_2023_little_kmers_World/' # define the path of directory
dir_week ='/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_little_spike2sig_World/' # define the path of directory


# column Variant --> labels
#metadata = pd.read_csv('/mnt/resources/2022_04/2022_04/filtered_metadata_0328_weeks.csv')
metadata = pd.read_csv('/blue/salemi/share/varcovid/SECONDO_ANNO/filtered_metadatataset_Nov2023_edit_221223_World.csv') # read the metadata file (CSV)  # leggo il file che devo salavare dove creo il dtaaset
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
path_save_file= '/blue/salemi/share/varcovid/SECONDO_ANNO/Pipeline/Spike2Sig/AIME/Official_simulation/nu_0_5/12Trimester/' # where save the file
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
df_trainstep_1 = df_trainstep_1[0].str.split(',', expand=True)
df_trainstep_1 = df_trainstep_1.fillna(0)
train_step1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()
train_step1 = truncate_sequences(train_step1, 1230)
# Converting the truncated sequences to integer lists
train_step1 = [[int(numero) for numero in sublist] for sublist in train_step1]
y_train_initial = metadata[metadata[col_lineage_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_lineage]  # Elements of training set.
lineages_train = np.array(y_train_initial.tolist())

n_training = 27
for i in range(2,n_training):
    print(i)
    df_trainstep_1, train_w_list = load_data(dir_week, [i]) # First training set.
    df_trainstep_1 = df_trainstep_1[0].str.split(',', expand=True)
    df_trainstep_1 = df_trainstep_1.fillna(0)
    train_step_1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()
    train_step_1 = truncate_sequences(train_step_1, 1230)
    # Converting the truncated sequences to integer lists
    train_step_1 = [[int(numero) for numero in sublist] for sublist in train_step_1]
    y_train_initial = metadata[metadata[col_lineage_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_lineage]  # Elements of training set.
    lineages_class = np.array(y_train_initial.tolist())  # Type of lineages.
    train_step1 = np.concatenate((train_step1 ,train_step_1))  # (rw = retraining week)
    lineages_train = np.concatenate((lineages_train,lineages_class))

print('End read the file')
# Control if length is the same
print(len(train_step1))
print(len(lineages_train))

ind_scaling = 0
train_completo = train_step1
# Finding the length of the shortest sequence
min_length = 1230

# Truncating all sequences to the minimum length
train_trunc = truncate_sequences(train_step1, min_length)

# Converting the truncated sequences to integer lists
train = [[int(numero) for numero in sublist] for sublist in train_trunc]

# training one class
clf_c = OneClassSVM(kernel='rbf', gamma='scale', nu=0.5,shrinking=True)
clf_c.fit(train)

# Creating Dictionary
y_test_dict_finalclass = {}
y_test_dict_predictedclass = {}
train_completo = train

# create the first file
test_step_tot = np.ones((1,1230))
y_test_nparray_fc = []
for week in range(n_training-1, 54):  # metadata['week'].max()
    logging.info("# Week " + str(starting_week + week))

    # Loading first test step
    df_teststep_i, test_w_list = load_data(dir_week, [starting_week + week])
    df_teststep_i = df_teststep_i[0].str.split(',', expand=True)
    df_teststep_i = df_teststep_i.fillna(0)
    test_step_i = df_teststep_i.iloc[:, 1:len(df_teststep_i.columns)].to_numpy()
    test_step_info = df_teststep_i.iloc[:, 1:len(df_teststep_i.columns)].to_numpy()
    test_step_i = truncate_sequences(test_step_i, min_length)
    test_step_i = [[int(numero) for numero in sublist] for sublist in test_step_i]
    test_step_completo = test_step_i

    # Class
    y_test_step_i = get_variant_class(metadata, df_teststep_i.iloc[:, 0].tolist())
    y_test_fclass_i = map_variant_to_finalclass(y_test_step_i, valid_lineage)
    i_voc = np.where(np.array(y_test_fclass_i) == -1)[0]
    lineages_type = metadata[metadata[col_lineage_id].isin(df_teststep_i.iloc[:, 0].tolist())][col_class_lineage]  # Type of lineages in the week of simulation.
    lineages_test = np.array(lineages_type.tolist())
    y_test_nparray_fc = np.concatenate((y_test_nparray_fc, np.array(y_test_fclass_i)))

    # Store the embedding
    train_completo = np.concatenate((train_completo, test_step_completo)) # contain all sequence
    lineages_train = np.concatenate((lineages_train, lineages_test))

    # store te test
    test_step_tot = np.concatenate((test_step_tot, test_step_completo))

# predict
y_test_i_predict = clf_c.predict(test_step_tot[1:])

# Update the dictionary
y_test_dict_finalclass[1] = list(y_test_nparray_fc)
y_test_dict_predictedclass[1] = y_test_i_predict

# saving results for this comb of param of the oneclass_svm
results = {'y_test_final_class': y_test_dict_finalclass,
       'y_test_predicted_class': y_test_dict_predictedclass}

results_fine_tune.append(results)
#PCA
lineages_train = np.array(transform_labels(lineages_train,valid_lineage_lineage))
plot_pca_with_centroids(train_completo,lineages_train,path_save_file)
plot_pca_with_centroids_zoomed(train_completo,lineages_train,path_save_file)

#Tsne
plot_tsne_with_centroids(train_completo,lineages_train,path_save_file)
plot_tsne_with_centroids_zoomed(train_completo,lineages_train,path_save_file)

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
df_conf.to_csv(path_save_file + 'conf_mat_over_time_s2s_ocsvm.tsv', sep='\t', index=None)