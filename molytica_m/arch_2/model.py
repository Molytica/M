import torch
from torch_geometric.nn import GATConv
from transformers import BertModel, AutoConfig, AutoModel
from torch import nn

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        return self.gat(x, edge_index)

# Assuming BERTLayer is a placeholder for ChemBERTa or similar
class BERTLayer(nn.Module):
    def __init__(self, model_name):
        super(BERTLayer, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output

class DenseLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseLayer, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)
    
class RegressionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super(RegressionHead, self).__init__()
        self.regression = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        return self.regression(x)


class Arch2Model(nn.Module):
    def __init__(self, gat_in_channels, gat_out_channels, chembert_model_name, protbert_model_name, dense_in_features, dense_out_features, regression_out_features):
        super(Arch2Model, self).__init__()
        self.gnn1 = GATLayer(gat_in_channels, gat_out_channels)
        self.gnn2 = GATLayer(gat_in_channels, gat_out_channels)
        # Initialize BERT layers with their specific pre-trained models
        self.chembert = BERTLayer(chembert_model_name)
        self.protbert = BERTLayer(protbert_model_name)
        self.dense1 = DenseLayer(dense_in_features, dense_out_features)
        self.dense2 = DenseLayer(dense_in_features, dense_out_features)
        # Replace the final dense layer with a regression head
        self.regression_head = RegressionHead(dense_out_features * 6, regression_out_features)

    def forward(self, data1, data2, input_ids1, attention_mask1, input_ids2, attention_mask2, x1, x2):
        gnn_out1 = self.gnn1(data1)
        gnn_out2 = self.gnn2(data2)
        bert_out1 = self.bert1(input_ids1, attention_mask1)
        bert_out2 = self.bert2(input_ids2, attention_mask2)
        dense_out1 = self.dense1(x1)
        dense_out2 = self.dense2(x2)
        combined = torch.cat((gnn_out1, gnn_out2, bert_out1, bert_out2, dense_out1, dense_out2), dim=1)
        return self.regression_head(combined)
    

def main():
    # Define channels/features for each layer
    gat_in_channels = 64
    gat_out_channels = 32
    dense_in_features = 768  # Size of the BERT layer output
    dense_out_features = 50  # Arbitrary number for intermediate dense layer
    regression_out_features = 2  # Replace with the actual number of regression targets

    # Define model names for the specific BERT models
    chembert_model_name = "DeepChem/ChemBERTa-77M-MLM"
    protbert_model_name = "Rostlab/prot_bert"

    # Initialize the model with names for both BERT models
    model = Arch2Model(gat_in_channels, gat_out_channels, chembert_model_name, protbert_model_name, dense_in_features, dense_out_features, regression_out_features)

    print(model)



gat_in_channels = 64
gat_out_channels = 32
dense_in_features = 768  # Size of the BERT layer output
dense_out_features = 50  # Arbitrary number for intermediate dense layer
regression_out_features = 2  # Replace with the actual number of regression targets

# Define model names for the specific BERT models
chembert_model_name = "DeepChem/ChemBERTa-77M-MLM"
protbert_model_name = "Rostlab/prot_bert"

def main():
    # Define channels/features for each layer
    
    # Initialize the model with names for both BERT models
    model = Arch2Model(gat_in_channels, gat_out_channels, chembert_model_name, protbert_model_name, dense_in_features, dense_out_features, regression_out_features)

    print(model)


def get_model():
    return Arch2Model(gat_in_channels, gat_out_channels, chembert_model_name, protbert_model_name, dense_in_features, dense_out_features, regression_out_features)

if __name__ == "__main__":
    main()