from molytica_m.data_tools.graph_tools import graph_tools
from molytica_m.target_selector import target_selector_tools
import sys

#interesting_uniprots = ["Q01860", "Q06416", "P48431", "O43474"] #OSK (ADD SITR1?)
interesting_uniprots = ["Q01860"]
model = "gpt-4-1106-preview"
search_tree = [15, 10, 3]
temperature = 0.0
n_rep_avg = 1

with open("molytica_m/target_selector/therapeutic_goal.txt", "r") as file:
    therapeutic_goal = file.read()

edges_to_evaluate = graph_tools.get_edges_from_tree(search_tree, interesting_uniprots)
#nodes_to_evaluate = graph_tools.get_nodes_from_tree(search_tree, interesting_uniprots)
#print(nodes_to_evaluate)
print(edges_to_evaluate)
sys.exit(0)
user_approval = True if input("Cost Will be projected to {} USD for this analysis. Proceed? y/n + Enter:".format(2 * len(edges_to_evaluate) * n_rep_avg * target_selector_tools.get_cost_for_one_eval(model))) == "y" else False
if not user_approval:
    print("Okay. Then aborting. ")
    sys.exit(0)
print("Proceeding with analysis.")
json_edges = target_selector_tools.evaluate_edges(edges_to_evaluate, temperature=temperature, therapeutic_goal=therapeutic_goal, model=model, n_rep_avg=n_rep_avg, interesting_uniprot_ids=interesting_uniprots, search_tree=search_tree, file_name="molytica_m/target_selector/GPT4_edges.json")
json_nodes = target_selector_tools.evaluate_nodes(nodes_to_evaluate, temperature=temperature, therapeutic_goal=therapeutic_goal, model=model, n_rep_avg=n_rep_avg, interesting_uniprot_ids=interesting_uniprots, search_tree=search_tree, file_name="molytica_m/target_selector/GPT4_nodes.json")