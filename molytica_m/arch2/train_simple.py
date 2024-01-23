from molytica_m.chembl_curation import get_chembl_data
from molytica_m.arch2 import chemBERT
from tqdm import tqdm
import torch, sqlite3, os, h5py
from molytica_m.arch2.simple_model import QSAR_model
import torch.optim as optim
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CV_split = get_chembl_data.get_categorised_CV_split()

train_data = []
for train_set in CV_split[:4]:
    train_data += train_set

test_data = []
for test_set in CV_split[4:]:
    test_data += test_set

def one_hot_encode(label):
    # Define the range of labels
    label_range = [-3, -2, -1, 0, 1, 2, 3]
    
    # Initialize a tensor of zeros with length equal to the number of unique labels
    one_hot = torch.zeros(len(label_range))
    
    # Find the index of the label in the label_range list
    label_index = label_range.index(label)
    
    # Set the corresponding position in the one_hot tensor to 1
    one_hot[label_index] = 1
    
    return one_hot.to(device)

def check_db_row_count(conn):
    """
    Check if the number of rows in the database is the same as the number of entries in the id_to_smiles dictionary.
    """
    query = "SELECT COUNT(*) FROM molecule_embeddings"
    c = conn.cursor()
    c.execute(query)
    row_count = c.fetchone()[0]
    return row_count

def get_molecule_embedding(smiles):
    """
    Retrieve the molecule embedding for the given SMILES string from the database.
    If the embedding does not exist, generate, store, and return it.
    """
    # Connect to the database
    conn = sqlite3.connect("data/curated_chembl/smiles_embeddings.db")
    cursor = conn.cursor()

    # Check if the embedding already exists
    cursor.execute("SELECT * FROM molecule_embeddings WHERE smiles = ?", (smiles,))
    row = cursor.fetchone()


    if row:
        # If the embedding exists, extract and return it
        embedding = row[2:]  # Assuming embedding starts from 4th column
        conn.close()
        return embedding
    else:
        # If embedding does not exist, generate it
        tok, mod = chemBERT.get_chemBERT_tok_mod()
        embed = chemBERT.get_molecule_mean_logits(smiles, tok, mod)[0][0]
        embed_array = embed.cpu().numpy()
        embed_list = embed_array.tolist()

        # Insert the new embedding into the database
        placeholders = ', '.join(['?'] * (2 + len(embed_list)))
        mol_id = check_db_row_count(conn) + 1
        sql_command = f"INSERT INTO molecule_embeddings (mol_id, smiles, {', '.join(['real' + str(i) for i in range(1, len(embed_list) + 1)])}) VALUES ({placeholders})"
        data_tuple = (mol_id, smiles,) + tuple(embed_list)
        cursor.execute(sql_command, data_tuple)
        conn.commit()
        conn.close()

        # Return the new embedding
        return embed_list


def load_protein_embedding(uniprot_id):
    for species in os.listdir(os.path.join("data", "curated_chembl", "protein_embeddings")):
        file_name = os.path.join("data", "curated_chembl", "protein_embeddings", species, f"{uniprot_id}_embedding.h5")
        if not os.path.exists(file_name):
            continue

        with h5py.File(file_name, 'r') as h5file:
            embedding = h5file['embedding'][()]
        
        return embedding

    print(f"File not found for UniProt ID {uniprot_id}")
    return None

model = QSAR_model(len(get_molecule_embedding("CC(=O)Oc1ccccc1C(=O)O")) + len(load_protein_embedding("Q01860")))
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

data_count = np.array([0, 0, 0, 0, 0, 0, 0])
data_dist = np.array([0, 0, 0, 0, 0, 0, 0])
SMA_n_100_acc = 0

progress_bar = tqdm(train_data, desc="Training")
for sample in progress_bar:
    optimizer.zero_grad()

    smiles = sample[0]
    mol_embed = torch.tensor(np.array(get_molecule_embedding(smiles)), dtype=torch.float32).unsqueeze(0).to(device)
    uniprot_id = sample[1]
    prot_embed = torch.tensor(np.array(load_protein_embedding(uniprot_id)), dtype=torch.float32).unsqueeze(0).to(device)
    label = torch.tensor(np.array(one_hot_encode(sample[6])), dtype=torch.float32).unsqueeze(0).to(device)

    predict = model(mol_embed, prot_embed)
    loss = torch.nn.functional.mse_loss(predict, label)
    loss.backward()
    optimizer.step()

    # Convert predictions and labels to class indices for comparison
    _, predicted_label = torch.max(predict.data, 1)
    _, true_label = torch.max(label.data, 1)

    # Increment the count for the correct label
    label_index = true_label.item()  # Assuming true_label is a single-element tensor
    data_count[label_index] += 1

    # Calculate the data distribution
    data_dist = data_count / data_count.sum()

    # Find the maximum value in data_dist
    max_value = data_dist.max()

    # Check if the prediction is correct (1 if correct, 0 if incorrect)
    correct_prediction = 1 if predicted_label == true_label else 0

    # Update the SMA with the correct prediction flag
    SMA_n_100_acc = 0.99 * SMA_n_100_acc + 0.01 * correct_prediction

    # Update the tqdm progress bar description
    progress_bar.set_description(f'Accuracy: {SMA_n_100_acc:.4f}, Max Dist Value: {max_value:.4f}')



# After training
model_save_path = "molytica_m/arch2/simple_QSAR.pth"  # Specify your path here
torch.save(model, model_save_path)
print(f"Complete model saved to {model_save_path}")


# Set the model to evaluation mode
model.eval()

test_data_count = np.array([0, 0, 0, 0, 0, 0, 0])
total_correct_predictions = 0
total_samples = 0

# Initialize tqdm progress bar
progress_bar = tqdm(test_data, desc='Evaluating')

for sample in progress_bar:
    smiles = sample[0]
    mol_embed = torch.tensor(np.array(get_molecule_embedding(smiles)), dtype=torch.float32).unsqueeze(0).to(device)
    uniprot_id = sample[1]
    prot_embed = torch.tensor(np.array(load_protein_embedding(uniprot_id)), dtype=torch.float32).unsqueeze(0).to(device)
    label = torch.tensor(np.array(one_hot_encode(sample[6])), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():  # No gradient calculation
        predict = model(mol_embed, prot_embed)

    # Convert predictions and labels to class indices for comparison
    _, predicted_label = torch.max(predict.data, 1)
    _, true_label = torch.max(label.data, 1)

    # Increment the count for the correct label
    label_index = true_label.item()
    test_data_count[label_index] += 1

    # Update correct predictions count
    total_correct_predictions += (predicted_label == true_label).sum().item()
    total_samples += label.size(0)

    # Calculate the current total accuracy
    total_accuracy = 100 * total_correct_predictions / total_samples

    # Update tqdm description with current total accuracy
    progress_bar.set_description(f'Total Accuracy: {total_accuracy:.2f}%')

print(f'Final Total Accuracy over Test Data: {total_accuracy:.2f}%')

