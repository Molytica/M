import sqlite3, sys
from molytica_m.data_tools import graph_tools
from molytica_m.data_tools import create_PPI_dataset
from torch_geometric.data import Data, Batch
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replace 'your_chembl_database_file.db' with the path to your ChEMBL SQLite file
database_path = 'data/chembl/smiles_alphafold_v4_human_uniprot_chembl_bioactivities.db'

# Connect to the SQLite database
conn = sqlite3.connect(database_path)

# Create a cursor object
cursor = conn.cursor()
batch_size = 30
n_IC50_EC50 = 878560
n_batches = int(n_IC50_EC50 / batch_size) + 1



try:
    for x in range(0, n_IC50_EC50, batch_size):
        # Execute a query (example: selecting 10 rows from a table)
        query = "SELECT * FROM bioactivities WHERE activity_unit = 'nM' AND (activity_type = 'IC50' OR activity_type = 'EC50') LIMIT {} OFFSET {};".format(batch_size, x)  # Replace 'your_table_name' with the actual table name
        cursor.execute(query)

        prot_metadatas = []
        IC50s = []
        EC50s = []
        mol_metadatas = []
        mol_Gs = []
        prot_Gs = []

        # Fetch and print the results
        rows = cursor.fetchall()
        for row in rows:
            smiles = row[0]
            uniprot_id = row[1]
            IC50 = 39900.0 #90th percentile value (high = low inhibition potency)
            EC50 = 26302.4 #90th percentile value (high = low effect potency)
            mol_metadata = [element for element in row[6:] if type(element) != str]

            if row[2] == 'IC50':
                IC50 = row[4]
            else:
                EC50 = row[4]

            mol_features, mol_csr_matrix = graph_tools.get_raw_graph_from_smiles_string(smiles)
            prot_features, prot_edge_index = create_PPI_dataset.get_graph_raw(uniprot_id)

            mol_edge_index_list = np.vstack(mol_csr_matrix.nonzero())

            mol_metadatas.append(mol_metadata)
            prot_metadatas.append(create_PPI_dataset.get_metadata(uniprot_id))

            mol_Gs.append(Data(x=torch.Tensor(mol_features), edge_index=torch.tensor(mol_csr_matrix, dtype=torch.long)))
            prot_Gs.append(Data(x=torch.Tensor(prot_features), edge_index=torch.tensor(prot_edge_index, dtype=torch.long)))


        metadata_as = np.array(metadata_as)
        metadata_bs = np.array(metadata_bs)
        metadata_as = torch.tensor(metadata_as, dtype=torch.float).to(device)
        metadata_bs = torch.tensor(metadata_bs, dtype=torch.float).to(device)

        batch_molss = Batch.from_data_list(mol_Gs).to(device)
        batch_prots = Batch.from_data_list(prot_Gs).to(device)



        
        print(rows)
        sys.exit(0)
except Exception as e:
    print(e)

# Close the connection
conn.close()