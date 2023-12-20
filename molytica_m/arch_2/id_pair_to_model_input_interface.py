
def get_model_input_interface(smiles, uniprot_id):
    gat_in_channels = 64
    gat_out_channels = 128
    chembert_model_name = "chembert-base"
    protbert_model_name = "protbert-base"
    dense_in_features = 256
    dense_out_features = 128
    regression_out_features = 2
    
    return gat_in_channels, gat_out_channels, chembert_model_name, protbert_model_name, dense_in_features, dense_out_features, regression_out_features
