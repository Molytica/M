from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

def get_protT5_stuff():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

    return tokenizer, model, device

def calculate_mean_embeddings(sequence_list, tokenizer, model, device):
    # Preprocess sequences
    processed_sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_list]

    # Tokenize and pad sequences
    ids = tokenizer(processed_sequences, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # Generate embeddings
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    mean_embeddings = []
    for i in range(len(sequence_list)):
        seq_len = len(sequence_list[i]) # Subtract special tokens
        emb = embedding_repr.last_hidden_state[i, :seq_len]
        emb_per_protein = emb.mean(dim=0)
        mean_embeddings.append(emb_per_protein)

    return mean_embeddings