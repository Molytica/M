from molytica_m.data_tools.graph_tools import graph_tools
from spektral.data import DisjointLoader
from spektral.data import Dataset
from tqdm import tqdm
import random
import os

class CustomDataset(Dataset):
    def __init__(self, data_folder, **kwargs):
        self.data_folder = data_folder
        super().__init__(**kwargs)

    def read(self):
        graphs = []

        graph_files = list(sorted(os.listdir(self.data_folder)))
        random.shuffle(graph_files)

        for graph_file in tqdm(graph_files, desc=f"Loading dataset from {self.data_folder}", unit="graphs"):
            graphs.append(graph_tools.load_graph(os.path.join(self.data_folder, graph_file)))

        return graphs

def get_disjoint_loader(data_folder, batch_size, epochs=None):
    return DisjointLoader(CustomDataset(data_folder), batch_size=batch_size, **{ 'epochs': epochs } if epochs is not None else {})