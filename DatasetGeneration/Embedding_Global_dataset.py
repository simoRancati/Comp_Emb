import statistics as st
from sklearn.model_selection import train_test_split
from utils import *

Continenti = ['World']
for l in Continenti:
    print('starting reading files')
    # Read the File csv
    df = pd.read_csv("/blue/salemi/share/varcovid/DeepAutoCov/metadata/metadata.csv") #
    sequences, headers = read_protein_sequences_header("/blue/salemi/share/varcovid/DeepAutoCov/spikeprot1105/spikes.fasta")
    df.iloc[:, 4] = df.iloc[:, 4].astype(type(headers[0]))
    df_ordinato = pd.DataFrame(headers, columns=['ID']).merge(df, left_on='ID', right_on=df.columns[4])
    df_ordinato['Sequences'] = sequences
    df = df_ordinato.drop('ID', axis=1)
    print('I have read the files')

    column_stratification = "Pango lineage"
    counter_classes = df[column_stratification].value_counts()
    remove_classes = counter_classes[counter_classes < 2].index
    df_filtered = df[~df[column_stratification].isin(remove_classes)]
    reduction = 2_000_000 / 4_000_000
    _, df_official, _, _ = train_test_split(df_filtered, df_filtered[column_stratification],
                                            stratify=df_filtered[column_stratification], test_size=reduction,
                                            random_state=42)

    sequences = df_official["Sequences"].values.tolist()
    df_official = df_official.drop(['Sequences'], axis=1)
    metadata = df_official.to_numpy()
    print('End Stratification')

    # Read the file
    print("\033[1m Read the metadata File \033[0m")
    metadata_nation = metadata

    # In the Fasta format, the asterisk is used to indicate the end of a sequence. The ending asterisk is removed in some sequences.
    print("\033[1mRemoval of the final asterisk in the Fasta format.\033[0m")
    sequences_nation = [remove_asterisks(s) for s in sequences]

    Dimension = len(sequences_nation)
    print('\033[1mNumber of spike protein is: ' + str(Dimension) + '\033[0m')
    # Lunghezza delle sequenze
    Length = []
    for sequence in sequences_nation:
        Length.append(len(sequence))

    print('\033[1mFilter sequences with length < 1000 \033[0m')
    sequences_filtered_min_1000 = [x for i, x in enumerate(sequences_nation) if Length[i] >= 1000]
    index_1 = [i for i, x in enumerate(Length) if x >= 1000]
    print('\033[1mAggiorno il file Metadata\033[0m')
    metadata_filtered_min_1000 = metadata_nation[index_1]
    Dimension_fil_min_1000 = len(sequences_filtered_min_1000)
    print('The number of spike proteins after eliminating sequences with length less than 1000 is: ' + str(Dimension_fil_min_1000))

    print('\033[1mCalculation of filtered lengths less than 1000\033[0m')
    Length_filtered_min_1000 = []
    for sequence in sequences_filtered_min_1000:
        Length_filtered_min_1000.append(len(sequence))

    # Seleziono le Sequenze Valide
    print('\033[1mEvaluate the valid sequences contained in the database\033[0m')
    valid_sequences, invalid_sequences, valid_indices, invalid_indices = validate_sequences(sequences_filtered_min_1000)
    print('\033[1mEvaluate : \033[0m')
    print('There are '+str(len(valid_sequences))+' valid sequences in database')
    print('There are '+str(len(invalid_sequences))+' invalid sequences in the database')

    print('\033[1mupdate filtered \033[0m')
    metadata_valid_indices = metadata_filtered_min_1000[valid_indices]
    metadata_valid_invalid_indices = metadata_filtered_min_1000[invalid_indices]

    print('\033[1mCalculate the new length of the filtered and valid sequences \033[0m')
    Length_filtered_min_1000_valid = []
    for sequence in valid_sequences:
        Length_filtered_min_1000_valid.append(len(sequence))

    print('\033[1mFilter sequences by the length within the median\033[0m')
    extreme_up = st.median(Length_filtered_min_1000_valid) - 30
    extreme_low = st.median(Length_filtered_min_1000_valid) + 30
    index_1,sequences_valid = filter_sequences(valid_sequences, extreme_up, extreme_low)
    metadata_valid_indices_length = metadata_valid_indices[index_1]
    print('Selected ' + str(len(sequences_valid)) + ' with length between ' + str(extreme_up) + ' and ' + str(extreme_low))

    print("\033[1mFilter sequences by dates \033[0m")
    metadata_off,sequences_off, metadata_not_off, sequences_not_off = filter_row_by_column_length_sostitution(metadata_valid_indices_length, sequences_valid, 5, 10) #Valuto che le date siano valide
    print("\033[1mThe number of sequences filtered with dates is :\033[0m"+str(len(metadata_off)))

    print("\033[1mReorder metadata file \033[0m")
    metadata_tot = insert_sequence_as_column(metadata_off,metadata_off[:,5],sequences_off)

    sequences = list(metadata_tot[:,24])
    metadata = metadata_tot[:,0:23]

    print('weeks')
    indices_by_week = split_weeks(metadata[:,5])
    print(len(indices_by_week))
    seq_week=[]
    for i in range(0,len(indices_by_week)):
        seq_week.append(len(indices_by_week[i]))
    print(seq_week)

    write_csv_dataset(metadata, l)

    print('\033[1mCompute k-mers\033[0m')
    k = 3
    kmers = calculate_kmers(sequences_valid, k)
    kmers_unici = list(set(kmers))

    for i in range(0,len(indices_by_week)):
        indices = indices_by_week[i]
        sequences_for_week = []
        identifier_for_week = []
        week = i+1
        # Creating Dataset
        os.makedirs('/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_kmers_' + l + '/' + str(week)) # mettiamolo al secondo anno
        os.makedirs('/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_spike2vec_'+l + '/' + str(week)) # mettiamolo al secondo anno
        os.makedirs('/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_spike2sig_'+l + '/' + str(week)) # mettiamolo al secondo anno
        os.makedirs('/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_fasta_'+l + '/' + str(week)) # mettiamolo al secondo anno
        for m,index in enumerate(indices): # qua ho le sequenze per settimana
            sequences_for_week.append(sequences[index])
            identifier_for_week.append(metadata[index,4])
            create_multiple_fasta_file(identifier_for_week, sequences_for_week, 'sequences.fasta', week, l)
        for h,seq in enumerate(sequences_for_week):
            format_csv(seq, identifier_for_week[h], kmers_unici, k, week, l) # kmers
            format_csv_spike2vec(seq, identifier_for_week[h], kmers_unici, k, week, l) #Spike2vec
            format_csv_spike2signal(seq, identifier_for_week[h], week, l) #Spike2signal

    # Creo il formato txt
    import os
    import csv
    # Spike2vec
    csv_directory = '/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_spike2vec_'+l

    # Loop attraverso tutte le sottodirectory e file nella cartella principale
    for root, directories, files in os.walk(csv_directory):
        for directory in directories:
            # Crea il file di testo e apri in modalità append
            txt_file = os.path.join(root, directory, "week_dataset.txt")
            with open(txt_file, "a") as output_file:
                for filename in os.listdir(os.path.join(root, directory)):
                    if filename.endswith(".csv"):
                        csv_file = os.path.join(root, directory, filename)
                        # Apri il file CSV con la libreria csv e leggi ogni riga
                        with open(csv_file, "r") as input_file:
                            reader = csv.reader(input_file)
                            next(reader)  # salta la prima riga
                            for row in reader:
                                # Scrivi ogni riga nel file di testo
                                output_file.write(",".join(row) + "\n")

    # Spike2sig
    csv_directory = '/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_spike2sig_'+l

    # Loop attraverso tutte le sottodirectory e file nella cartella principale
    for root, directories, files in os.walk(csv_directory):
        for directory in directories:
            # Crea il file di testo e apri in modalità append
            txt_file = os.path.join(root, directory, "week_dataset.txt")
            with open(txt_file, "a") as output_file:
                for filename in os.listdir(os.path.join(root, directory)):
                    if filename.endswith(".csv"):
                        csv_file = os.path.join(root, directory, filename)
                        # Apri il file CSV con la libreria csv e leggi ogni riga
                        with open(csv_file, "r") as input_file:
                            reader = csv.reader(input_file)
                            next(reader)  # salta la prima riga
                            for row in reader:
                                # Scrivi ogni riga nel file di testo
                                output_file.write(",".join(row) + "\n")
    # Baseline
    csv_directory = '/blue/salemi/share/varcovid/SECONDO_ANNO/dataset_nov_2023_kmers_' + l

    # Loop attraverso tutte le sottodirectory e file nella cartella principale
    for root, directories, files in os.walk(csv_directory):
        for directory in directories:
            # Crea il file di testo e apri in modalità append
            txt_file = os.path.join(root, directory, "week_dataset.txt")
            with open(txt_file, "a") as output_file:
                for filename in os.listdir(os.path.join(root, directory)):
                    if filename.endswith(".csv"):
                        csv_file = os.path.join(root, directory, filename)
                        # Apri il file CSV con la libreria csv e leggi ogni riga
                        with open(csv_file, "r") as input_file:
                            reader = csv.reader(input_file)
                            next(reader)  # salta la prima riga
                            for row in reader:
                                # Scrivi ogni riga nel file di testo
                                output_file.write(",".join(row) + "\n")