from molytica_m.data_tools.alpha_fold_tools import get_alphafold_uniprot_ids
from molytica_m.data_tools.graph_tools import graph_tools
from molytica_m.data_tools.graph_tools import interactome_tools
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import json
import os
    
