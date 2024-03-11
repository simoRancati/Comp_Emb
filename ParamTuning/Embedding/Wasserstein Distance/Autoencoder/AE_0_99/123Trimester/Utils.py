from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import numpy as np
from datetime import *
from keras import layers, models, losses

def load_data(dir_dataset, week_range):
    week_range = [str(x) for x in week_range]
    weeks_folder = [x for x in os.listdir(dir_dataset) if x in week_range]
    df_list = []
    w_list = []

    for week in weeks_folder:
        week_path = os.path.join(dir_dataset, week)
        for root, dirs, files in os.walk(week_path):
            for file in files:
                if file.endswith(".fasta"):
                    fasta_path = os.path.join(root, file)
                    with open(fasta_path, 'r') as f:
                        # Qui leggiamo il file FASTA e lo convertiamo in un formato DataFrame
                        header, sequence = '', ''
                        sequences = []
                        for line in f:
                            if line.startswith('>'):
                                if header:
                                    sequences.append([header, sequence])
                                header = line.strip()
                                sequence = ''
                            else:
                                sequence += line.strip()
                        # Aggiungi l'ultima sequenza
                        if header:
                            sequences.append([header, sequence])

                        df = pd.DataFrame(sequences, columns=['Header', 'Sequence'])
                        df['Header'] = df['Header'].str.replace('>', '', regex=False)
                        df_list.append(df)
                        w_list += [week] * df.shape[0]

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

def find_min_length(sequences):
    """
    Finds the length of the shortest sequence in the list.
    """
    return min(len(seq) for seq in sequences)

def truncate_sequences(sequences, target_length):
    """
    Truncates each sequence in the list to the specified target length.
    """
    return [seq[:target_length] for seq in sequences]

# Amino acids (standard 20)
amino_acids = 'ACDEFGHIKLMNPQRSTVWY' # Standard Aminoacid
aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}

def safe_convert_to_int(s):
    try:
        return int(s)
    except ValueError:
        return s  # Return the original string if it can't be converted to an integer

# One-hot encoding for amino acids
def one_hot_encode(aa):
    idx = aa_to_idx[aa]
    vector = torch.zeros(len(amino_acids))
    vector[idx] = 1
    return vector

# Custom dataset for protein sequences
class ProteinDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoded_sequence = torch.stack([one_hot_encode(aa) for aa in sequence])
        return encoded_sequence

# Simple neural network for feature extraction
class ProteinFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim):
        super(ProteinFeatureExtractor, self).__init__()
        self.fc = nn.Linear(len(amino_acids), embedding_dim)

    def forward(self, x):
        # Average over the sequence length (dimension 0)
        x = torch.mean(x, dim=0)
        return self.fc(x)

# Wasserstein distance function
def wasserstein_distance(embeddings1, embeddings2):
    return torch.norm(embeddings1 - embeddings2, p=1)

# Training function
def train_protein_network_and_extract_embeddings(dataset, embedding_dim, epochs, learning_rate, batch_size):
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = ProteinFeatureExtractor(embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Store for embeddings
    embeddings_store = []

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            optimizer.zero_grad()

            # Forward pass to get embeddings
            embeddings = model(data.float())

            # Store embeddings
            embeddings_store.append(embeddings.detach().cpu().numpy())

            # Wasserstein distance (placeholder)
            loss = wasserstein_distance(embeddings, embeddings)  # Needs modification

            # Backward pass
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    return embeddings_store

def get_embedding_for_multiple_sequences(model, sequences):
    model.eval()  # Set the model to evaluation mode
    embeddings = []

    with torch.no_grad():
        for sequence in sequences:
            # Convert the sequence to one-hot encoding
            encoded_sequence = torch.stack([one_hot_encode(aa) for aa in sequence])

            # Get embedding for each sequence
            embedding = model(encoded_sequence.float())
            embeddings.append(list(embedding))

    return embeddings

def convert_tensors_to_lists(tensor_lists):
    return [[tensor.item() for tensor in tensor_list] for tensor_list in tensor_lists]

def lineages_of_interest():
    valid_lineage_FDLs = ['B.1.2', 'B.1.1.7', 'AY.4', 'B.1.617.2', 'AY.29', 'AY.103','AY.43', 'AY.44', 'BA.2', 'BA.1', 'BA.2.9', 'BA.2.12.1', 'BA.5', 'BA.5.1', 'BA.5.2', 'BR.2.1', 'XBB.1.5','CH.1.1', 'FK.1.1', 'XBC.1.6', 'XBC.1.3', 'DV.7.1', 'HW.1.1', 'EG.5.1']
    valid_lineage = valid_lineage_FDLs

    return valid_lineage

def retraining_weeks():
    retraining_week = [11, 12, 30, 32, 41, 45, 53, 77, 87, 104, 105, 110,124, 137, 140, 153, 163, 167, 182, 185, 192, 195, 197]
    return retraining_week

def flatten_list_of_lists(list_of_lists):
    return [sublist[0] for sublist in list_of_lists if sublist]

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

def plot_pca_with_centroids(embeddings, labels, path_save):
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
    plt.savefig(path_save + "pca_centroid.png", bbox_inches='tight')
    plt.show()


def plot_pca_with_centroids_zoomed(embeddings, labels,path_save):
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
    plt.savefig(path_save + "pca_centroid_zoom.png", bbox_inches='tight')
    plt.show()

def plot_tsne_with_centroids_zoomed(embeddings, labels,path_save):
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
    plt.savefig(path_save + "tsne_centroid_zoom.png", bbox_inches='tight')
    plt.show()

def plot_tsne_with_centroids(embeddings, labels, path_save):
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
    plt.savefig(path_save + "tsne_centroid.png", bbox_inches='tight')
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

def build_noisy_autoencoder(input_dim=30, latent_dim=16, noise_factor=0.05):
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(encoder_inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(latent_dim, activation='relu')(x)

    # Aggiunta del rumore gaussiano
    noise = layers.GaussianNoise(noise_factor)(x)

    # Decoder
    x = layers.Dense(64, activation='relu')(noise)
    x = layers.Dense(128, activation='relu')(x)
    decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)  # Adjust activation based on your data

    # Autoencoder
    autoencoder = models.Model(encoder_inputs, decoder_outputs)

    return autoencoder

def compute_reconstruction_error(model, data):
    reconstructions = model.predict(data)
    mse = np.mean(np.square(data - reconstructions), axis=1)
    return mse

def define_anomaly_threshold(errors, quantile=0.98):
    # Definisce la soglia come il quantile specificato degli errori di ricostruzione sui dati di training
    threshold = np.quantile(errors, quantile)
    return threshold

def detect_anomalies(errors, threshold):
    # Usa un'operazione vettoriale per confrontare ogni errore con la soglia
    # Ritorna -1 per anomalie e 1 per dati normali
    return np.where(errors > threshold, -1, 1)