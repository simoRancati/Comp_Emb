# Main
from Utils import *
import logging
from sklearn.linear_model import SGDOneClassSVM
import numpy as np

# WEEK DIRECTORY
# dir_week ='/Users/utente/Desktop/Varcovid/Comparing_Different_Embedding/dataset_nov_2023_little_fasta_World' # define the path of directory
dir_week = '/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_little_fasta_World/'  # path del Dataset #path del Dataset

# column Variant --> labels
#metadata = pd.read_csv('/mnt/resources/2022_04/2022_04/filtered_metadata_0328_weeks.csv')
metadata = pd.read_csv('/blue/salemi/share/varcovid/SECONDO_ANNO/filtered_metadatataset_Nov2023_edit_221223_World.csv')  # leggo il file che devo salavare dove creo il dtaaset
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

"""
Useful variable

"""
#path_salvataggio_file='/mnt/resources/2022_04/2022_04/'
path_save_file= '/blue/salemi/share/varcovid/SECONDO_ANNO/Pipeline/Weistress_Distance/AIME/Official_simulation/nu_0_01/1Trimester/' # where save the file
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
train_step1 = flatten_list_of_lists(train_step1)
lineages_train = np.array(y_train_initial.tolist())
# Finding the length of the shortest sequence
min_length = 1230
# Truncating all sequences to the minimum length
train_step1 = truncate_sequences(train_step1, min_length)
# Converting the truncated sequences to integer lists
train_step1 = [[safe_convert_to_int(numero) for numero in sublist] for sublist in train_step1]
lineages_train = np.array(y_train_initial.tolist())

n_training = 14
for i in range(2,n_training):
    print(i)

    df_trainstep_1, train_w_list = load_data(dir_week, [i]) # First training set.
    train_step_1 = df_trainstep_1.iloc[:, 1:len(df_trainstep_1.columns)].to_numpy()
    train_step_1 = flatten_list_of_lists(train_step_1)
    # Finding the length of the shortest sequence
    min_length = 1230
    # Truncating all sequences to the minimum length
    train_step_1 = truncate_sequences(train_step_1, min_length)
    # Converting the truncated sequences to integer lists
    train_step_1 = [[safe_convert_to_int(numero) for numero in sublist] for sublist in train_step_1]

    y_train_initial = metadata[metadata[col_lineage_id].isin(df_trainstep_1.iloc[:, 0].tolist())][col_class_lineage]  # Elements of training set.
    lineages_class = np.array(y_train_initial.tolist())  # Type of lineages.
    train_step1 = np.concatenate((train_step1 ,train_step_1))  # (rw = retraining week)
    lineages_train = np.concatenate((lineages_train,lineages_class))

print('End read the file')
# Control if length is the same
print(len(train_step1))
print(len(lineages_train))

ind_scaling = 0

# Finding the length of the shortest sequence
min_length = 1230

# Truncating all sequences to the minimum length
train_trunc = truncate_sequences(train_step1, min_length)

# Converting the truncated sequences to integer lists
train = [[safe_convert_to_int(numero) for numero in sublist] for sublist in train_trunc]

# Reduce fraction

# Calcola la dimensione desiderata del dataset ridimensionato
desired_size = 7000 if len(lineages_train) > 7000 else int(len(lineages_train) * 0.1)

# Calcola le frequenze delle categorie
unique, counts = np.unique(lineages_train, return_counts=True)
frequencies = dict(zip(unique, counts))

# Calcola le proporzioni per ciascuna categoria
total = len(lineages_train)
proportions = {k: v / total for k, v in frequencies.items()}

# Seleziona gli indici per ciascuna categoria
selected_indices = []
for category, proportion in proportions.items():
    # Calcola il numero di esempi da selezionare per questa categoria
    n_samples = int(proportion * desired_size)

    # Ottieni gli indici per tutti gli esempi di questa categoria
    category_indices = np.where(np.array(lineages_train) == category)[0]

    # Se non ci sono abbastanza esempi, usa replace=True per il campionamento
    replace = n_samples > len(category_indices)

    # Campiona gli indici per questa categoria
    sampled_indices = np.random.choice(category_indices, size=n_samples, replace=replace)

    # Aggiungi gli indici selezionati alla lista finale
    selected_indices.extend(sampled_indices)

# Se necessario, campiona ulteriormente per ottenere esattamente la dimensione desiderata
if len(selected_indices) > desired_size:
    selected_indices = np.random.choice(selected_indices, size=desired_size, replace=False)

Train_wd = [train[i] for i in selected_indices]
## Waistrass embedding
dataset = ProteinDataset(Train_wd)
train_protein_network_and_extract_embeddings(dataset, embedding_dim=32, epochs=50, learning_rate=0.001, batch_size=64)
model = ProteinFeatureExtractor(embedding_dim=32)
embedding = get_embedding_for_multiple_sequences(model, train)
train = convert_tensors_to_lists(embedding)

print('ho finito di allenare wd')
# training one class
clf_c = SGDOneClassSVM(nu = 0.01,random_state=42)
clf_c.fit(train)

# Dictionary
y_test_dict_finalclass = {}
y_test_dict_predictedclass = {}
train_completo = train # contain the sequence with all the kmers

# define some varibles
test_step_tot = np.ones((1,32))
y_test_nparray_fc = []

for week in range(n_training-1, 54):  # metadata['week'].max()
    logging.info("# Week " + str(starting_week + week))

    # Loading first test step
    df_teststep_i, test_w_list = load_data(dir_week, [starting_week + week])
    test_step_i = df_teststep_i.iloc[:, 1:len(df_teststep_i.columns)].to_numpy()
    test_step_i = flatten_list_of_lists(test_step_i)
    test_step_i = truncate_sequences(test_step_i, min_length)
    test_step_i= [[safe_convert_to_int(numero) for numero in sublist] for sublist in test_step_i]
    embedding = get_embedding_for_multiple_sequences(model, test_step_i)
    test_step_i = convert_tensors_to_lists(embedding)
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

    # Store the Test
    test_step_tot = np.concatenate((test_step_tot, test_step_completo))

# predict
# Returns -1 for outliers and 1 for inliers.
y_test_i_predict = clf_c.predict(test_step_tot[1:])
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
df_conf.to_csv(path_save_file + 'conf_mat_over_time_wd_ocsvm_GSD.tsv', sep='\t', index=None)

