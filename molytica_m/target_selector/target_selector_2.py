from molytica_m.data_tools.graph_tools import graph_tools
from molytica_m.target_selector import target_selector_tools
import sys

interesting_uniprots = ["Q01860", "Q06416", "P48431", "O43474"] #OSK (ADD SITR1?)
#model = "gpt-4-1106-preview
model = "gpt-3.5-turbo-1106"
temperature = 0.0
n_rep_avg = 1
budget = 1 # USD
n_samples = int(budget / target_selector_tools.get_cost_for_one_eval(model))
fraction_node_samples = 0.5
fraction_edge_samples = 1 - fraction_node_samples

with open("molytica_m/target_selector/therapeutic_goal.txt", "r") as file:
    therapeutic_goal = file.read()

# Sample n_samples edges
