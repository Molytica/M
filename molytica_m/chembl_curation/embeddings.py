
import torch, re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/prot_t5_xl_uniref50")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_protein_embeddings(uniprot_ids):
    uniprot_sequences = {}

    for uniprot_id in uniprot_ids:
        sequence = load_protein_sequence(uniprot_id)
        if sequence:
            # Replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            processed_sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            uniprot_sequences[uniprot_id] = processed_sequence


    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(uniprot_sequences.values(), add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)

    # extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7]) 
    for i, uniprot_id in tqdm(enumerate(uniprot_sequences.keys()), total=len(uniprot_sequences), desc="Processing embeddings"):
        emb = embedding_repr.last_hidden_state[i,:len(uniprot_sequences[uniprot_id].replace(" ", ""))]
        emb_per_protein = emb.mean(dim=0)  # shape (1024)

        with h5py.File(os.path.join("data/curated_chembl/af_protein_embeddings", f"{uniprot_id}_embeddings.h5"), 'w') as h5file:
            h5file.create_dataset('embeddings', data=np.array(emb_per_protein, dtype=float))

def read_protein_embedding(uniprot_id, directory="data/curated_chembl/af_protein_embeddings"):
    file_path = os.path.join(directory, f"{uniprot_id}_embeddings.h5")
    
    if not os.path.exists(file_path):
        print(f"File not found for UniProt ID {uniprot_id}")
        return None

    try:
        with h5py.File(file_path, 'r') as h5file:
            # Assuming the dataset is named 'embeddings'
            if 'embeddings' in h5file:
                embeddings = np.array(h5file['embeddings'])
                return embeddings
            else:
                print(f"'embeddings' dataset not found in {file_path}")
                return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None
