import sqlite3
from tqdm import tqdm
from molytica_m.data_tools import id_mapping_tools

def filter_tid(tid):
    if id_mapping_tools.tid_to_af_uniprot(tid):
        return True
    return False

# Path to the original database
original_db_path = 'data/chembl/chembl_33.db'

# Path to the new database
new_db_path = 'data/chembl/filtered_data.db'

# Connect to the original database
conn_original = sqlite3.connect(original_db_path)
cursor_original = conn_original.cursor()

# Connect to the new database (this will create the file if it doesn't exist)
conn_new = sqlite3.connect(new_db_path)
cursor_new = conn_new.cursor()

# Create a new table in the new database
cursor_new.execute("""
CREATE TABLE IF NOT EXISTS filtered_data (
    canonical_smiles VARCHAR(4000),
    af_4_human_uniprot VARCHAR(10),
    activity_type VARCHAR(250),
    activity_relation VARCHAR(50),
    activity_value NUMERIC,
    activity_unit VARCHAR(100),
    molregno BIGINT,
    mw_freebase NUMERIC(9, 2),
    alogp NUMERIC(9, 2),
    hba INTEGER,
    hbd INTEGER,
    psa NUMERIC(9, 2),
    rtb INTEGER,
    ro3_pass VARCHAR(3),
    num_ro5_violations SMALLINT,
    cx_most_apka NUMERIC(9, 2),
    cx_most_bpka NUMERIC(9, 2),
    cx_logp NUMERIC(9, 2),
    cx_logd NUMERIC(9, 2),
    molecular_species VARCHAR(50),
    full_mwt NUMERIC(9, 2),
    aromatic_rings INTEGER,
    heavy_atoms INTEGER,
    qed_weighted NUMERIC(3, 2),
    mw_monoisotopic NUMERIC(11, 4),
    full_molformula VARCHAR(100),
    hba_lipinski INTEGER,
    hbd_lipinski INTEGER,
    num_lipinski_ro5_violations SMALLINT,
    np_likeness_score NUMERIC(3, 2)
);
""")

total_rows = 6749321  # Adjust based on the total number of rows

for x in tqdm(range(0, total_rows, 1000000), desc="loading"):
    query = f"""
    SELECT compound_structures.canonical_smiles, assays.tid, activities.standard_type, activities.standard_relation, activities.standard_value, activities.standard_units, compound_properties.*
    FROM activities 
    JOIN assays ON activities.assay_id = assays.assay_id
    JOIN compound_structures ON activities.molregno = compound_structures.molregno
    JOIN compound_properties ON activities.molregno = compound_properties.molregno
    WHERE assays.assay_organism = 'Homo sapiens'
    LIMIT 1000000 OFFSET {x};
    """
    cursor_original.execute(query)
    rows = cursor_original.fetchall()
    
    for row in rows:
        if filter_tid(row[1]):  # Replace tid_column_index with the index of tid in the row
            new_row = list(row)
            new_row[1] = id_mapping_tools.tid_to_af_uniprot(new_row[1])

            cursor_new.execute("INSERT INTO filtered_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(new_row))  # Adjust the placeholders based on the number of columns

# Commit changes to the new database and close both connections
conn_new.commit()
conn_original.close()
conn_new.close()
