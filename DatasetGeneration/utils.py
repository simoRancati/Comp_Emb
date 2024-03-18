# Function
import pandas as pd
import csv
import os
import numpy as np
from datetime import datetime
import random
import csv

def read_csv(file):
    return pd.read_csv(file).values
def read_tsv(file):
    return pd.read_csv(file,sep='\t').values

def read_fasta(file):
    sequences = []
    with open(file, 'r') as f:
        current_sequence = ''
        started = False
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if started:
                    sequences.append(current_sequence)
                started = True
                current_sequence = ''
            else:
                current_sequence += line
        if current_sequence:
            sequences.append(current_sequence)
    return sequences

def read_protein_sequences_header(file):
    """
    This function reads a FASTA file, extracts protein sequences, and their respective headers.

    Parameters:
    - file: The path to the FASTA file.

    Returns:
    - sequences: A list containing only the protein sequences found in the FASTA file.
    - headers: A list containing the headers for each protein sequence.
    """

    sequences = []  # Initialize an empty list to store protein sequences.
    headers = []    # Initialize an empty list to store headers of the sequences.

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        current_sequence = ''  # A variable to hold the current sequence.

        for line in f:
            line = line.strip()  # Strip whitespace from the line.

            if line.startswith('>'):  # Check for header line.
                if current_sequence:  # If there's a current sequence, store it before starting a new one.
                    sequences.append(current_sequence)
                    current_sequence = ''

                # Add the header line to the headers list, excluding the first character ('>')
                headers.append(line[1:])

            else:
                current_sequence += line  # Append non-header lines to the current sequence.

        if current_sequence:  # After the last line, add the final sequence.
            sequences.append(current_sequence)

    return sequences, headers

def validate_sequences(sequences):
    valid_sequences = []
    invalid_sequences = []
    valid_indices = []
    invalid_indices = []
    for index, seq in enumerate(sequences):
        is_valid = True
        for amino_acid in seq:
            if amino_acid not in "ACDEFGHIKLMNPQRSTVWY":
                is_valid = False
                break
        if is_valid:
            valid_sequences.append(seq)
            valid_indices.append(index)
        else:
            invalid_sequences.append(seq)
            invalid_indices.append(index)
    return valid_sequences, invalid_sequences, valid_indices, invalid_indices

def remove_asterisks(sequence):
    return sequence.rstrip("*")

def filter_sequences(sequenze, lunghezza_minima, lunghezza_massima): # La lunghezza massima e minima la ecidiamo noi cioè io metto la mediana generale - 20 amminoacidi e + 20 amminoacidi
    indici = [i for i, seq in enumerate(sequenze) if lunghezza_minima <= len(seq) <= lunghezza_massima]
    sequenze_valide = [seq for i, seq in enumerate(sequenze) if lunghezza_minima <= len(seq) <= lunghezza_massima]
    return indici, sequenze_valide

def spike2signal(seq_data):
    total_int_seq = []

    for i in range(len(seq_data)):
        dnaSeq = list(seq_data[i])
        res = [item.replace('A', '11') for item in dnaSeq]
        res = [item.replace('C', '10') for item in res]
        res = [item.replace('D', '9') for item in res]
        res = [item.replace('E', '8') for item in res]
        res = [item.replace('F', '7') for item in res]
        res = [item.replace('G', '6') for item in res]
        res = [item.replace('H', '5') for item in res]
        res = [item.replace('I', '4') for item in res]
        res = [item.replace('K', '3') for item in res]
        res = [item.replace('L', '2') for item in res]
        res = [item.replace('M', '1') for item in res]
        res = [item.replace('N', '-1') for item in res]
        res = [item.replace('P', '-2') for item in res]
        res = [item.replace('Q', '-3') for item in res]
        res = [item.replace('R', '-4') for item in res]
        res = [item.replace('S', '-5') for item in res]
        res = [item.replace('T', '-6') for item in res]
        res = [item.replace('V', '-7') for item in res]
        res = [item.replace('W', '-8') for item in res]
        res = [item.replace('X', '-9') for item in res]
        res = [item.replace('Y', '-10') for item in res]

    data = []
    for i in range(len(res)):
        data.append(float(res[i]))
    # print(data)

    total_int_seq.append(data)

    com_sum_full_data = []

    for i in range(len(total_int_seq)):
        com_sum_full_data.append(np.cumsum(total_int_seq[i]))  # cumulative sum

    def float_to_int(lista_float):
        return [int(x) for x in lista_float]


    for i in range(len(com_sum_full_data)):
        list_signal = list(com_sum_full_data[i])
        int_signal = float_to_int(list_signal)

    return int_signal

# example of use
#signal= spike2signal(['ACCDFRRTWWXY'])

def calculate_kmers(sequences, k):
    kmers = []
    for sequence in sequences:
        for i in range(len(sequence)-k+1):
            kmer = sequence[i:i+k]
            kmers.append(kmer)
    return kmers

def format_csv(seq,identificativo,kmers_tot,k,week,l):
    # poi le dovremmo passare lineage,VOC
    kmers=[]
    binary=[]
    binary.append(identificativo)
    kmers.append(None)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        kmers.append(kmer)
    for i,km in enumerate(kmers_tot):
        if kmers_tot[i] in kmers:
            binary.append(1)
        else:
            binary.append(0)
    kmers_tot=[None]+kmers_tot
    #os.makedirs('/Users/utente/Desktop/Varcovid/Nuovi_dati/'+str(week))
    with open('/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_kmers_' + l + '/'+str(week)+'/'+str(identificativo)+'.csv', 'w', newline='') as file: # to change
        writer = csv.writer(file)
        writer.writerow(kmers_tot)
        writer.writerow(binary)
    return 'done'

def format_csv_spike2vec(seq, identificativo, kmers_tot, k, week, l):
    # Contare l'occorrenza di ciascun k-mer nella sequenza
    def count_kmers(sequence, k):
        kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
        counts = {}
        for kmer in kmers:
            if kmer in counts:
                counts[kmer] += 1
            else:
                counts[kmer] = 1
        return counts

    kmers_counts = count_kmers(seq, k)

    kmers_tot = [None] + kmers_tot
    binary = [identificativo]

    # Sostituisco il controllo di presenza/assenza con il conteggio reale
    for km in kmers_tot[1:]:  # saltare il valore None
        binary.append(kmers_counts.get(km, 0))

    #os.makedirs('/Users/utente/Desktop/Varcovid/PAPER_GROSSO/Nuovi_dati/'+str(week)+'/dataset_febb_2023_' + l + '/' + str(week))
    with open('/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_spike2vec_'+l+'/'+str(week)+'/'+str(identificativo)+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(kmers_tot)
        writer.writerow(binary)

    return 'done'

def format_csv_spike2signal(seq, identificativo, week,l):
    # Contare l'occorrenza di ciascun k-mer nella sequenza
    seq = [seq]
    signal = spike2signal(seq)

    first_row = list(range(1, len(seq) + 1))
    first_row_tot = [None] + first_row

    second_row = [identificativo] + signal
    #os.makedirs('/Users/utente/Desktop/Varcovid/PAPER_GROSSO/Nuovi_dati/'+str(week)+'/dataset_febb_2023_' + l + '/' + str(week))
    with open('/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_spike2sig_'+l + '/'+str(week)+'/'+str(identificativo)+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_row_tot)
        writer.writerow(second_row)  # Scrivi la sequenza nella seconda riga

    return 'done'

def create_multiple_fasta_file(identifiers, sequences, file_name,week,l):
    """
    Creates a FASTA file with multiple sequences and identifiers.

    :param identifiers: List of sequence identifiers.
    :param sequences: List of corresponding sequences.
    :param file_name: Name of the FASTA file to create.
    """
    with open('/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_fasta_'+l + '/'+str(week)+'/'+str(file_name), "w") as file:
        for identifier, sequence in zip(identifiers, sequences):
            file.write(f">{identifier}\n")
            for i in range(0, len(sequence), 80):
                file.write(sequence[i:i+80] + "\n")


def filter_row_by_column_length(ndarray, string_list, col_index, target_len):
    mask = np.array([len(s) == target_len for s in ndarray[:,col_index]], dtype=bool)
    return ndarray[mask], [s for s, m in zip(string_list, mask) if m], ndarray[np.logical_not(mask)], [s for s, m in zip(string_list, mask) if not m]

def filter_row_by_column_length_sostitution(ndarray, string_list, col_index, target_len):

    def check_and_add_suffix(s):
        suffixes = ["-01", "-07", "-10", "-14", "-20", "-25", "-27"]
        if len(s) == 7:
            extended_s = s + random.choice(suffixes)
            if len(extended_s) == target_len:
                return extended_s
        return s

    extended_strings = np.array([check_and_add_suffix(s) for s in ndarray[:, col_index]])
    mask = np.array([len(s) == target_len for s in extended_strings], dtype=bool)
    ndarray[:, col_index] = extended_strings

    return ndarray[mask], [s for s, m in zip(string_list, mask) if m], ndarray[np.logical_not(mask)], [s for s, m in zip(string_list, mask) if not m]

def insert_sequence_as_column(data, dates, sequence):
    """
    Inserisce la sequenza di aminoacidi come colonna di un ndarray che contiene i metadati,
    ordinando poi l'ndarray in base alle date (in ordine crescente).

    Args:
        data (ndarray): l'ndarray contenente i metadati.
        dates (list): la lista delle date corrispondenti a ciascuna riga dell'ndarray.
        sequence (list): la lista della sequenza di aminoacidi da inserire come colonna.

    Returns:
        ndarray: l'ndarray ordinato in base alle date, con la sequenza di aminoacidi inserita come colonna.
    """
    # Trasforma le date in oggetti datetime
    date_objs = np.array([datetime.strptime(date, '%Y-%m-%d') for date in dates])

    # Aggiungi la colonna della sequenza come ultima colonna dell'ndarray
    data_with_sequence = np.column_stack((data, sequence))

    # Aggiungi una colonna con gli oggetti datetime delle date
    data_with_dates = np.column_stack((data_with_sequence, date_objs))

    # Ordina l'ndarray in base alle date
    sorted_indices = np.argsort(data_with_dates[:, -1])
    sorted_data = data_with_dates[sorted_indices]

    # Elimina la colonna delle date
    sorted_data = np.delete(sorted_data, -1, axis=1)

    return sorted_data

def select_rows(ndarray,country):
    selected_rows = []
    for row in ndarray:
        if isinstance(row[6], str) and country in row[6]:
            selected_rows.append(row)
    return np.array(selected_rows)

# to create dataset
def select_rows_dataset(ndarray, country):
    selected_rows = []
    selected_indices = []

    for index, row in enumerate(ndarray):
        if isinstance(row[6], str) and country in row[6]:
            selected_rows.append(row)
            selected_indices.append(index)

    return np.array(selected_rows), selected_indices

def split_weeks(dates):
    date_objs = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    date_objs.sort()
    start_date = date_objs[0]
    end_date = date_objs[-1]

    num_weeks = ((end_date - start_date).days // 7) + 1
    indices_by_week = [[] for _ in range(num_weeks)]

    for i, date_obj in enumerate(date_objs):
        days_diff = (date_obj - start_date).days
        week_num = days_diff // 7
        indices_by_week[week_num].append(i)

    return indices_by_week

def trimestral_indices(dates_list,m):
    # Converti le date in oggetti datetime
    dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in dates_list]

    # Crea un dizionario che associa a ogni trimestre (anno, trimestre) la lista degli indici delle date in quel trimestre
    trimestral_indices = {}
    for i, date in enumerate(dates):
        year = date.year
        trimester = (date.month - 1) // m + 1
        key = (year, trimester)
        if key not in trimestral_indices:
            trimestral_indices[key] = []
        trimestral_indices[key].append(i)

    # Restituisci la lista di liste degli indici dei trimestri, ordinati per anno e trimestre
    sorted_keys = sorted(trimestral_indices.keys())
    return [trimestral_indices[key] for key in sorted_keys]

def write_csv_dataset(array,l):
    # Definizione dei nomi delle colonne come lista di stringhe
    nomi_colonne = ['Virus.name','Not.Impo','format','Type','Accession.ID','Collection.date','Location','Additional.location.information','Sequence.length','Host','Patient.age','Gender','Clade','Pango.lineage','Pangolin.type','Variant','AA.Substitutions','Submission.date','Is.reference.','Is.complete.','Is.high.coverage.','Is.low.coverage.','N.Content']
    # Apertura del file CSV in modalità scrittura e definizione del writer
    with open('/blue/salemi/share/varcovid/SECONDO_ANNO/filtered_metadatataset_Nov2023_edit_221223_'+l+'.csv', "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        # Scrittura della riga d'intestazione con i nomi delle colonne
        writer.writerow(nomi_colonne)

        # Scrittura delle righe dei dati
        for riga in array:
            writer.writerow(riga)