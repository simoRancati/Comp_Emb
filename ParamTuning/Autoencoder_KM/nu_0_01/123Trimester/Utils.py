import pandas as pd
import os
import numpy as np
from datetime import *
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf

def load_data(dir_dataset, week_range):
    week_range = [str(x) for x in week_range]
    weeks_folder = [x for x in os.listdir(dir_dataset) if x in week_range]
    df_list = []
    w_list = []
    for week in weeks_folder:
        df_path = dir_dataset  + week +'/week_dataset.txt'
        df = pd.read_csv(df_path, header=None)
        # df = df[~df.iloc[:, 0].isin(id_unknown)]
        df_list.append(df)
        w_list += [week]*df.shape[0]
    return pd.concat(df_list), w_list

def get_variant_class(metadata, id_list):
    variant_name_list = []
    for id in id_list:
        variant_name_list.append(metadata[metadata['Accession.ID'] == id]['Pango.lineage'].values[0])
    return variant_name_list

def map_variant_to_finalclass(class_list, non_neutral):
    # -1 -> non-neutral
    # 1 -> neutral
    final_class_list = []
    for c in class_list:
        if c in non_neutral:
            final_class_list.append(-1)
        else:
            final_class_list.append(1)
    return final_class_list

def get_time(date_string, date_format="%Y-%m-%d"):
    return datetime.strptime(date_string, date_format).date()

def build_deep_autoencoder(list_length, encoding_dim):
    # Input shape adjusted for the length of the binary list
    input_shape = (list_length,)

    # Input layer
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # No need to flatten since the input is already a 1D list

    # Encoder layers
    encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)

    # Decoder layers
    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    # Adjusted output layer to match the size of the input list
    decoded = tf.keras.layers.Dense(list_length, activation='sigmoid')(decoded)

    # Autoencoder model
    autoencoder = tf.keras.models.Model(input_layer, decoded)

    # Encoder model
    encoder = tf.keras.models.Model(input_layer, encoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder


def lineages_of_interest():
    valid_lineage_FDLs = ['B.1.2', 'B.1.1.7', 'AY.4', 'B.1.617.2', 'AY.29', 'AY.103','AY.43', 'AY.44', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.5', 'BA.5.1', 'BA.5.2', 'BR.2.1', 'XBB.1.5','CH.1.1', 'FK.1.1', 'XBC.1.6', 'XBC.1.3', 'DV.7.1', 'HW.1.1', 'EG.5.1']
    valid_lineage = valid_lineage_FDLs

    return valid_lineage

def retraining_weeks():
    retraining_week = [11, 12, 30, 32, 41, 45, 53, 77, 87, 104, 105, 110,124, 137, 140, 153, 163, 167, 182, 185, 192, 195, 197]
    return retraining_week

def visualize_embeddings_with_pca(embeddings, labels, title="PCA Visualization of Embeddings"):
    """
    Visualizza gli embedding utilizzando PCA.

    :param embeddings: numpy.ndarray, i dati di embedding.
    :param labels: numpy.ndarray, le etichette delle classi (-1, 1).
    :param title: str, titolo del grafico.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Applicazione della PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Visualizzazione
    plt.figure(figsize=(8, 6))

    # Colori per le classi -1 e 1
    colors = ['red', 'green']

    # Disegna i punti per ogni classe
    for i, label in enumerate([-1, 1]):
        plt.scatter(embeddings_2d[labels == label, 0], embeddings_2d[labels == label, 1], c=colors[i], label=f'Class {label}')

    plt.title(title)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.show()

def visualize_ocsvm_with_pca(embeddings, ocsvm_model, title="PCA Visualization with One-Class SVM"):
    """
    Visualizza gli embedding utilizzando PCA e le classificazioni di un One-Class SVM.

    :param embeddings: numpy.ndarray, i dati di embedding.
    :param ocsvm_model: OneClassSVM, il modello One-Class SVM addestrato.
    :param title: str, titolo del grafico.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Applicazione della PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Esegui le previsioni del SVM sui dati
    predictions = ocsvm_model.predict(embeddings)

    # Visualizzazione
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=predictions, cmap='coolwarm', edgecolor='k')
    plt.title(title)
    plt.show()

def visualize_ocsvm_with_pca_Train(embeddings, ocsvm_model, week, path_save, title="PCA Visualization with One-Class SVM"):
    """
    Visualizza gli embedding utilizzando PCA e le classificazioni di un One-Class SVM.

    :param embeddings: numpy.ndarray, i dati di embedding.
    :param ocsvm_model: OneClassSVM, il modello One-Class SVM addestrato.
    :param week: int, settimana per il salvataggio del file.
    :param path_save: str, percorso per salvare il grafico.
    :param title: str, titolo del grafico.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Applicazione della PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Esegui le previsioni del SVM sui dati
    predictions = ocsvm_model.predict(embeddings)

    # Visualizzazione
    plt.figure(figsize=(18, 12))

    # Scatter plot per i punti normali e anomali
    plt.scatter(embeddings_2d[predictions == 1, 0], embeddings_2d[predictions == 1, 1], c='blue', edgecolor='k', label='Not Anomaly', s=100, alpha = 0.3)
    plt.scatter(embeddings_2d[predictions == -1, 0], embeddings_2d[predictions == -1, 1], c='red', edgecolor='k', label='Anomaly', s=100, alpha = 0.3)

    plt.title(title, fontsize=20)
    plt.xlabel("First Principal Component", fontsize=15)
    plt.ylabel("Second Principal Component", fontsize=15)
    plt.legend(prop={'size': 15})

    # Salvataggio dell'immagine senza bordi
    plt.savefig(path_save + "pca_ocsvm_visualization_Train_"+str(week)+".png", bbox_inches='tight')

    plt.show()
    return predictions, embeddings_2d, pca

def visualize_ocsvm_with_pca_Test(embeddings, new_embeddings, predictions, new_predictions, pca_model, week, path_save, title="PCA Visualization with One-Class SVM"):
    """
    Visualizza gli embedding originali e nuovi utilizzando PCA e le classificazioni di un One-Class SVM.

    :param embeddings: numpy.ndarray, i dati di embedding originali.
    :param new_embeddings: numpy.ndarray, i nuovi dati di embedding da aggiungere.
    :param predictions: numpy.ndarray, predizioni del modello SVM per i dati originali.
    :param new_predictions: numpy.ndarray, predizioni del modello SVM per i nuovi dati.
    :param pca_model: PCA, il modello PCA addestrato.
    :param week: int, settimana per il salvataggio del file.
    :param path_save: str, percorso per salvare il grafico.
    :param title: str, titolo del grafico.
    """
    import matplotlib.pyplot as plt

    # Applicazione della PCA sui dati originali
    # embeddings_2d = embeddings

    # Applicazione della PCA sui nuovi dati

    new_embeddings_2d = pca_model.fit_transform(new_embeddings)

    # Visualizzazione
    plt.figure(figsize=(18, 12))

    # Scatter plot per i punti originali
    #plt.scatter(embeddings_2d[predictions == 1, 0], embeddings_2d[predictions == 1, 1], c='blue', edgecolor='k', label='Not Anomaly (Train)', s=50, alpha = 0.3)
    #plt.scatter(embeddings_2d[predictions == -1, 0], embeddings_2d[predictions == -1, 1], c='red', edgecolor='k', label='Anomaly (Train)', s=50, alpha = 0.3)

    # Scatter plot per i nuovi punti
    plt.scatter(new_embeddings_2d[new_predictions == 1, 0], new_embeddings_2d[new_predictions == 1, 1], c='green', edgecolor='k', label='Not anomaly (Test)', s=50, alpha = 0.3)
    plt.scatter(new_embeddings_2d[new_predictions == -1, 0], new_embeddings_2d[new_predictions == -1, 1], c='orange', edgecolor='k', label='Anomaly (Test)', s=50, alpha = 0.3)

    plt.title(title, fontsize=20)
    plt.xlabel("First Principal Component", fontsize=15)
    plt.ylabel("Second Principal Component", fontsize=15)
    plt.legend(prop={'size': 15})

    # Salvataggio dell'immagine senza bordi
    plt.savefig(path_save + "pca_ocsvm_visualization_test_"+str(week)+'.png', bbox_inches='tight')

    plt.show()

def plot_pca_with_centroids(embeddings, labels):
    # Standardizzazione dei dati
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(embeddings)

    # Applicazione di PCA per ridurre a 2 dimensioni
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)

    # Preparazione del plot
    plt.figure(figsize=(10, 8))

    # Calcolo e plot dei centroidi per le classi note
    unique_labels = np.unique(labels)
    for cls in unique_labels:
        if cls != 'unknown':
            idxs = labels == cls
            centroid = np.mean(X_pca[idxs], axis=0)
            plt.plot(centroid[0], centroid[1], marker='o', markersize=10, label=f'Centroid {cls}', alpha = 0.4)

    # Visualizzazione della distribuzione degli 'Unknown' con KDE
    idxs_unknown = labels == 'unknown'
    if np.any(idxs_unknown):
        x_unknown, y_unknown = X_pca[idxs_unknown, 0], X_pca[idxs_unknown, 1]
        kde = gaussian_kde([x_unknown, y_unknown])
        xx, yy = np.meshgrid(np.linspace(x_unknown.min(), x_unknown.max(), 100),
                             np.linspace(y_unknown.min(), y_unknown.max(), 100))
        zz = kde(np.array([xx.flatten(), yy.flatten()]))
        plt.contourf(xx, yy, zz.reshape(xx.shape), levels=8, cmap='Oranges', alpha=0.3)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Projection with Class Centroids')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pca_with_centroids_zoomed(embeddings, labels):
    # Applicazione di PCA per ridurre a 2 dimensioni
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(embeddings)

    # Preparazione del plot
    plt.figure(figsize=(10, 8))

    # Calcolo dei centroidi per le classi note
    unique_labels = np.unique(labels)
    centroids = []
    for cls in unique_labels:
        if cls != 'unknown':
            idxs = labels == cls
            centroid = np.mean(X_pca[idxs], axis=0)
            centroids.append(centroid)
            plt.plot(centroid[0], centroid[1], marker='o', markersize=10, label=f'Centroid {cls}', alpha=0.4)

    # Visualizzazione della distribuzione degli 'Unknown' con KDE
    idxs_unknown = labels == 'unknown'
    if np.any(idxs_unknown):
        x_unknown, y_unknown = X_pca[idxs_unknown, 0], X_pca[idxs_unknown, 1]
        kde = gaussian_kde([x_unknown, y_unknown])
        xx, yy = np.meshgrid(np.linspace(x_unknown.min(), x_unknown.max(), 100),
                             np.linspace(y_unknown.min(), y_unknown.max(), 100))
        zz = kde(np.array([xx.flatten(), yy.flatten()]))
        plt.contourf(xx, yy, zz.reshape(xx.shape), levels=8, cmap='Oranges', alpha=0.3)

    # Impostazione degli assi per focalizzarsi sulla zona principale
    if centroids:
        centroids = np.array(centroids)
        mean_centroids = np.mean(centroids, axis=0)
        std_centroids = np.std(centroids, axis=0)

        plt.xlim(mean_centroids[0] - 10*std_centroids[0], mean_centroids[0] + 10*std_centroids[0])
        plt.ylim(mean_centroids[1] - 10*std_centroids[1], mean_centroids[1] + 10*std_centroids[1])

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Projection with Class Centroids - Zoomed In')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tsne_with_centroids_zoomed(embeddings, labels):
    # Applicazione di t-SNE per ridurre a 2 dimensioni
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
    X_tsne = tsne.fit_transform(embeddings)

    # Preparazione del plot
    plt.figure(figsize=(10, 8))

    # Calcolo dei centroidi per le classi note
    unique_labels = np.unique(labels)
    centroids = []
    for cls in unique_labels:
        if cls != 'unknown':
            idxs = labels == cls
            centroid = np.mean(X_tsne[idxs], axis=0)
            centroids.append(centroid)
            plt.plot(centroid[0], centroid[1], marker='o', markersize=10, label=f'Centroid {cls}', alpha=0.4)

    # Visualizzazione della distribuzione degli 'Unknown' con KDE
    idxs_unknown = labels == 'unknown'
    if np.any(idxs_unknown):
        x_unknown, y_unknown = X_tsne[idxs_unknown, 0], X_tsne[idxs_unknown, 1]
        kde = gaussian_kde([x_unknown, y_unknown])
        xx, yy = np.meshgrid(np.linspace(x_unknown.min(), x_unknown.max(), 100),
                             np.linspace(y_unknown.min(), y_unknown.max(), 100))
        zz = kde(np.array([xx.flatten(), yy.flatten()]))
        plt.contourf(xx, yy, zz.reshape(xx.shape), levels=8, cmap='Oranges', alpha=0.3)

    # Impostazione degli assi per focalizzarsi sulla zona principale
    if centroids:
        centroids = np.array(centroids)
        mean_centroids = np.mean(centroids, axis=0)
        std_centroids = np.std(centroids, axis=0)

        plt.xlim(mean_centroids[0] - 2*std_centroids[0], mean_centroids[0] + 2*std_centroids[0])
        plt.ylim(mean_centroids[1] - 2*std_centroids[1], mean_centroids[1] + 2*std_centroids[1])

    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('2D t-SNE Projection with Class Centroids - Zoomed In')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tsne_with_centroids(embeddings, labels):
    # Applicazione di t-SNE per ridurre a 2 dimensioni
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
    X_tsne = tsne.fit_transform(embeddings)

    # Preparazione del plot
    plt.figure(figsize=(10, 8))

    # Calcolo dei centroidi per le classi note
    unique_labels = np.unique(labels)
    centroids = []
    for cls in unique_labels:
        if cls != 'unknown':
            idxs = labels == cls
            centroid = np.mean(X_tsne[idxs], axis=0)
            centroids.append(centroid)
            plt.plot(centroid[0], centroid[1], marker='o', markersize=10, label=f'Centroid {cls}', alpha=0.4)

    # Visualizzazione della distribuzione degli 'Unknown' con KDE
    idxs_unknown = labels == 'unknown'
    if np.any(idxs_unknown):
        x_unknown, y_unknown = X_tsne[idxs_unknown, 0], X_tsne[idxs_unknown, 1]
        kde = gaussian_kde([x_unknown, y_unknown])
        xx, yy = np.meshgrid(np.linspace(x_unknown.min(), x_unknown.max(), 100),
                             np.linspace(y_unknown.min(), y_unknown.max(), 100))
        zz = kde(np.array([xx.flatten(), yy.flatten()]))
        plt.contourf(xx, yy, zz.reshape(xx.shape), levels=8, cmap='Oranges', alpha=0.3)

    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('2D t-SNE Projection with Class Centroids')
    plt.legend()
    plt.grid(True)
    plt.show()

def transform_labels(labels, parent_classes):
    transformed_labels = []
    for label in labels:
        # Gestione dei casi 'unknown' e simili
        if label in parent_classes or 'unknown' in label:
            transformed_labels.append(label)
            continue

        # Ricerca del padre più specifico per l'etichetta corrente
        parent_found = None
        for parent in parent_classes:
            if label.startswith(parent):
                # Seleziona il padre più lungo che corrisponde (più specifico)
                if parent_found is None or len(parent) > len(parent_found):
                    parent_found = parent

        # Aggiunta del padre trovato o dell'etichetta originale se non è stato trovato alcun padre
        transformed_labels.append(parent_found if parent_found else label)

    return transformed_labels


def generate_sublineages(lineages, sublineages_count=9):
    """
    Genera una lista di sublineages per ogni lineage fornito.

    Args:
    lineages (list): Lista dei lineages iniziali.
    sublineages_count (int): Numero di sublineages da generare per ogni lineage.

    Returns:
    list: Lista contenente i lineages originali e i loro sublineages generati.
    """
    all_lineages = []  # Lista per contenere sia i lineages originali che i sublineages
    for lineage in lineages:
        all_lineages.append(lineage)  # Aggiungi il lineage originale
        for i in range(1, sublineages_count + 1):
            sublineage = f"{lineage}.{i}"  # Genera il sublineage
            all_lineages.append(sublineage)  # Aggiungi il sublineage alla lista

    return all_lineages