import torch
import esm
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist, pdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# torch.hub.set_dir("/cs/usr/nil340/models")

# All of ESM-2 pre-trained models by embedding size
ESM_MODELS_DICT = {320: esm.pretrained.esm2_t6_8M_UR50D,
                   480: esm.pretrained.esm2_t12_35M_UR50D,
                   640: esm.pretrained.esm2_t30_150M_UR50D,
                   1280: esm.pretrained.esm2_t33_650M_UR50D,
                   2560: esm.pretrained.esm2_t36_3B_UR50D,
                   5120: esm.pretrained.esm2_t48_15B_UR50D}


def load_peptide_data(data_csv="DB/NesDB_all_CRM1_with_peptides_train.csv", include_nesdoubt=True,
                      include_nodoubt=True, max_peptide_len=22):
    """
    Loads all negative and positive peptide data
    :param data_csv: The path to the NesDB csv file
    :param include_nesdoubt: Whether or not to include doubt data (default=True)
    :param include_nodoubt: Whether or not to include no doubt data (default=True)
    :param max_peptide_len: Maximal allowed length for a peptide (default=22)
    :return: Three lists: [positive peptides], [negative peptides], [is doubt labels]
    """
    df = pd.read_csv(data_csv).dropna(subset=['Peptide_sequence', 'Negative_sequence', 'Sequence'])
    if not include_nesdoubt:
        df = df[df['is_NesDB_doubt'] != True].reset_index(drop=True)
    if not include_nodoubt:
        df = df[df['is_NesDB_doubt'] != False].reset_index(drop=True)

    pos_pep = []
    neg_pep = []
    data_doubt = []
    counter = 0

    for index, row in df.iterrows():
        pep = row['Peptide_sequence']
        neg = row['Negative_sequence']
        if len(pep) <= max_peptide_len and pep != '' and len(neg) <= max_peptide_len and neg != '':
            pos_pep.append((f"{counter}", pep))
            neg_pep.append((f"{counter}", neg))
            data_doubt.append(row['is_NesDB_doubt'])
            counter += 1

    return pos_pep, neg_pep, data_doubt


def get_esm_model(embedding_size=1280):
    """
    Retrieves a pre-trained ESM-2 model
    :param embedding_size: The ESM-2 model embedding size
    :return: esm_model, alphabet, batch_converter, device
    """

    if embedding_size not in ESM_MODELS_DICT:
        raise ValueError(f"ERROR: ESM does not have a trained model with embedding size of {embedding_size}.\n "
                         f"Please use one of the following embedding sized: {ESM_MODELS_DICT.keys()}")

    model, alphabet = ESM_MODELS_DICT[embedding_size]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    # check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"ESM model loaded to {device}")
    return model, alphabet, batch_converter, device


def get_esm_sequence_embedding(pep_tuple_list, esm_model, alphabet, batch_converter, device, embedding_layer=33):
    """
    This function convert peptide sequence data into ESM sequence embeddings
    :param pep_tuple_list: peptide tuple list of format : [(name_1, seq_1), (name_2, seq_2), ...]
    :param esm_model: Pre-trained ESM-2 model
    :param alphabet: ESM-2 alphabet object
    :param batch_converter: ESM-2 batch_converter object
    :param embedding_layer: The desired embedding layer to get
    :return: List of ESM-2 sequence embeddings
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(pep_tuple_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations
    with torch.no_grad():
        results = esm_model(batch_tokens.to(device), repr_layers=[embedding_layer])
    token_representations = results["representations"][embedding_layer]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(list(token_representations[i, 1: tokens_len - 1].mean(0).cpu().numpy()))

    return sequence_representations

def pep_train_test_split(pos_pep, neg_pep, doubt_list, test_size=0.1):
    """
    Splits the peptide data into training and testing sets
    :param pos_pep: positive peptides ESM-2 sequence embeddings
    :param neg_pep: negative peptides ESM-2 sequence embeddings
    :param doubt_list: doubt labels list
    :param test_size: Desired test size
    :return: pos_train, neg_train, doubt_train, pos_test, neg_test, doubt_test
    """
    assert len(pos_pep) == len(neg_pep) == len(doubt_list)
    pos_pep, neg_pep, doubt_list = np.array(pos_pep), np.array(neg_pep), np.array(doubt_list)

    # Split to train and test
    train_idx, test_idx = train_test_split(range(len(pos_pep)), test_size=test_size)
    pos_train, pos_test = pos_pep[train_idx], pos_pep[test_idx]
    neg_train, neg_test = neg_pep[train_idx], neg_pep[test_idx]
    doubt_train, doubt_test = doubt_list[train_idx], doubt_list[test_idx]

    return pos_train, neg_train, doubt_train, pos_test, neg_test, doubt_test


def get_peptide_distances(pos_neg_test_peptides, pos_train_peptides, reduce_func=np.min):
    """
    Returns for each peptide in 'pos_neg_test_peptides' the mean euclidean distance from all of the peptides in
     'pos_train_peptides'
    :param pos_neg_test_peptides: ESM-2 sequence embeddings of test peptides (negative or positives)
    :param pos_train_peptides: ESM-2 sequence embeddings of train peptides (positives)
    :param reduce_func: How to reduce the pair distances (mean/median...)
    :return: numpy array with distance for eact peptide in 'pos_neg_test_peptides'
    """
    # Get all of the Euclidean distances of the ESM embeddings for each test-train pair
    distances = cdist(np.array(pos_neg_test_peptides), np.array(pos_train_peptides), metric="euclidean")

    # For each test peptide get the mean/median/max.. etc. from all the positive training peptides
    reduced_distances = reduce_func(distances, axis=-1)

    return reduced_distances

def knn_classify_peptides(test_peptides, train_peptides, train_labels, k=2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_peptides, train_labels)
    return knn.predict_proba(test_peptides)[:, 1]

def plot_roc_curve(y_test, y_scores, label, color, lw=2):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=lw, label=f'{label} (AUC = {roc_auc:.2f})')

def plot_knn_roc_curves(y_test, y_scores_dict, out_file_path="knn_roc_curves.png"):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(y_scores_dict)))

    for (label, y_scores), color in zip(y_scores_dict.items(), colors):
        plot_roc_curve(y_test, y_scores, label, color)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('KNN ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_file_path)

def plot_other_roc_curves(y_test, y_scores_dict, out_file_path="all_models_roc_curves.png"):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(y_scores_dict)))

    for (label, y_scores), color in zip(y_scores_dict.items(), colors):
        plot_roc_curve(y_test, y_scores, label, color)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Other Models ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(out_file_path)

def plot_logreg_box_plot(y_test, y_scores_logreg, out_file_path="logistic_regression_boxplot.png"):
    # Separate the scores into positive and negative groups based on true labels
    positive_scores = [score for score, label in zip(y_scores_logreg, y_test) if label == 1]
    negative_scores = [score for score, label in zip(y_scores_logreg, y_test) if label == 0]

    # Combine scores into a list for plotting
    data_to_plot = [positive_scores, negative_scores]

    # Create the box plot
    fig, ax = plt.subplots()
    ax.boxplot(data_to_plot, tick_labels=['Positive', 'Negative'])

    # Add title and labels
    ax.set_title('Logistic Regression Predicted Probabilities')
    ax.set_xlabel('Class')
    ax.set_ylabel('Predicted Probability')

    # Show the plot
    plt.savefig(out_file_path)


def train_and_evaluate_models(chosen_embedding_size, chosen_embedding_layer, chosen_test_size):
    # Load all the peptide data
    print("Loading peptide data")
    positive_pep, negative_pep, doubt_lables = load_peptide_data()

    # Load the pre-trained ESM-2 model with the desired embedding size
    print("Loading ESM-2 model")
    model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=chosen_embedding_size)

    # Get the ESM-2 sequence embeddings fro all the negative and positive peptides
    print("Getting the ESM-2 embeddings for all the peptide data")
    positive_esm_emb = get_esm_sequence_embedding(positive_pep, model_esm, alphabet_esm, batch_converter_esm,
                                                  device_esm, embedding_layer=chosen_embedding_layer)
    negative_esm_emb = get_esm_sequence_embedding(negative_pep, model_esm, alphabet_esm, batch_converter_esm,
                                                  device_esm, embedding_layer=chosen_embedding_layer)

    # Split the data into train and test sets
    print("Splitting to train and test sets")
    positive_train, negative_train, is_doubt_train, positive_test, negative_test, is_doubt_test = pep_train_test_split(
        positive_esm_emb, negative_esm_emb, doubt_lables, test_size=chosen_test_size)

    # Labels for training data (1 for positive, 0 for negative)
    train_labels = [1] * len(positive_train) + [0] * len(negative_train)
    train_peptides = np.concatenate([positive_train, negative_train])
    test_peptides = np.concatenate([positive_test, negative_test])
    y_test = [1] * len(positive_test) + [0] * len(negative_test)

    # KNN classification and ROC curves
    knn_y_scores_dict = {}
    knn_best_auc = 0
    knn_best_k = 1

    for k in range(1, 16):
        y_scores_knn = knn_classify_peptides(test_peptides, train_peptides, train_labels, k=k)
        knn_y_scores_dict[f'KNN (k={k})'] = y_scores_knn

        # Determine best k based on AUC
        fpr, tpr, _ = roc_curve(y_test, y_scores_knn)
        roc_auc = auc(fpr, tpr)
        if roc_auc > knn_best_auc:
            knn_best_auc = roc_auc
            knn_best_k = k

    # Plot KNN ROC curves
    plot_knn_roc_curves(y_test, knn_y_scores_dict)

    # K-means classification and ROC curve
    print("Training K-means model with k=2")
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(train_peptides)
    y_scores_kmeans = kmeans.predict(test_peptides)
    y_scores_kmeans = 1 - y_scores_kmeans  # Adjust for ROC

    # Logistic Regression classification and ROC curve
    print("Training Logistic Regression model")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(train_peptides, train_labels)
    y_scores_logreg = logreg.predict_proba(test_peptides)[:, 1]

    # Cosine Similarity classification and ROC curve
    print("Calculating Cosine Similarity scores")
    cosine_sim = cosine_similarity(test_peptides, train_peptides)
    y_scores_cosine = np.array([cosine_sim[i].max() for i in range(len(cosine_sim))])

    # Best KNN model classification and ROC curve
    print(f"Using best KNN model with k={knn_best_k}")
    y_scores_best_knn = knn_y_scores_dict[f'KNN (k={knn_best_k})']

    # Plot Other ROC curves
    all_models_y_scores_dict = {
        'K-means (k=2)': y_scores_kmeans,
        'Logistic Regression': y_scores_logreg,
        'Cosine Similarity': y_scores_cosine,
        f'Best KNN (k={knn_best_k})': y_scores_best_knn
    }
    plot_other_roc_curves(y_test, all_models_y_scores_dict)

    # Box plot for logistic regression
    plot_logreg_box_plot(y_test, y_scores_logreg)

    knn = KNeighborsClassifier(n_neighbors=knn_best_k)

    all_models_dict = {'K-means (k=2)': kmeans,
                       'Logistic Regression': logreg,
                       'Cosine Similarity': cosine_sim,
                       f'Best KNN (k={knn_best_k})': knn}

    # Calculate AUC scores for each model
    auc_scores = {model: roc_auc_score(y_test, scores) for model, scores in all_models_y_scores_dict.items()}

    # Find the model with the best AUC score
    best_model_key = max(auc_scores, key=auc_scores.get)

    return all_models_dict[best_model_key], model_esm, alphabet_esm, batch_converter_esm, device_esm


def predict_peptide(peptide_sequence, best_model, model_esm, alphabet_esm, batch_converter_esm, device_esm):
    # Convert the peptide sequence to a tuple list for the batch converter
    pep_tuple_list = [("user_peptide", peptide_sequence)]

    # Get the ESM sequence embedding for the user peptide
    peptide_embedding = get_esm_sequence_embedding(pep_tuple_list, model_esm, alphabet_esm, batch_converter_esm,
                                                   device_esm)

    # Convert the list to numpy array and reshape it to match the model's expected input shape
    peptide_embedding = np.array(peptide_embedding).reshape(1, -1)

    # Make prediction using the best model
    prediction = best_model.predict(peptide_embedding)
    confidence = best_model.predict_proba(peptide_embedding)[0][prediction]
    confidence = round(confidence[0] * 100, 2)

    return prediction[0], confidence


if __name__ == '__main__':
    chosen_embedding_size = 1280
    chosen_embedding_layer = 33
    chosen_test_size = 0.1

    best_model, model_esm, alphabet_esm, batch_converter_esm, device_esm = (
        train_and_evaluate_models(chosen_embedding_size, chosen_embedding_layer, chosen_test_size))
    # User input
    user_peptide = input("Please enter peptide sequence or exit to stop: ")

    # Generating output
    while user_peptide != "exit":
        prediction, confidence = predict_peptide(user_peptide, best_model, model_esm, alphabet_esm,
                                                 batch_converter_esm, device_esm)
        if prediction == 1:
            print(f"The peptide is predicted to be positive with confidence of {confidence}%.")
        else:
            print(f"The peptide is predicted to be negative with confidence of {confidence}%")
        user_peptide = input("Please enter peptide sequence or exit to stop: ")