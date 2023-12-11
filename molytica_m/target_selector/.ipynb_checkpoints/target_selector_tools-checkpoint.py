from molytica_m.data_tools import PPI_Interactome_tools
from molytica_m.data_tools import alpha_fold_tools
from molytica_m.data_tools import gpt_tools
from openai import OpenAI
from tqdm.notebook import tqdm
import pandas as pd
import json

def get_uniprot_data(uniprot_id):
    df_idmappings = pd.read_table("molytica_m/data_tools/idmapping_2023_11_18.tsv")
    uniprot_information = df_idmappings.loc[df_idmappings["Entry"] == uniprot_id].values.tolist()
    return uniprot_information

def get_cost(input_chars, output_chars, model):
    input_tokens = input_chars / 4
    output_tokens = output_chars / 4
    if model == "gpt-3.5-turbo-1106":
        return 0.001 * input_tokens / 1000 + 0.002 * output_tokens / 1000
    if model == "gpt-4-1106-preview":
        return 0.01 * input_tokens / 1000 + 0.03 * output_tokens / 1000
    else:
        print("Please add model pricing in the get cost function for pricing.")

def get_cost_for_one_eval(model):
    if model == "gpt-4-1106-preview":
        return 1.35 / 15 / 3
    else:
        print("Please add model pricing in the get cost function for pricing.")

def evaluate_edge_helpfullness(edge, temperature, therapeutic_goal, n_rep_avg, model):
    iPPI_helpfullness = 0

    total_cost = 0
    reps = 0

    failed_attemps = 0
    while reps < n_rep_avg:
        try:
            gpt_input = "The first protein is\n\n\n" + str(get_uniprot_data(edge[0])) + "\n\n\nAND the second protein, which could the another protein of the same id, is:\n\n\n" + str(get_uniprot_data(edge[1])) + "\n\n\nMake your judgement and only respond with a number from -100 to 100 based on the instructions."
            gpt_prompt = "From -100 to 100. 100 = should inhibit because helps goal a lot and has no side effects, -100 = do not inhibit because does not help goal and has a lot of side effects. How helpful would inibiting the interaction between these proteins given by uniptors plus other descriptors be for achiving the following goal. Make an educated judgement using your comprehensive biological knowledge and systems understanding. The goal is as follows: {}".format(therapeutic_goal)
            gpt_evaluation = gpt_tools.ask_gpt(gpt_input, gpt_prompt, model, temperature)

            print(gpt_evaluation)
            total_cost += get_cost_for_one_eval(model)
        
            iPPI_helpfullness += float(gpt_evaluation) / 100.0 / float(n_rep_avg)
            reps += 1
        except Exception as e:
            failed_attemps += 1
            print(e)
            print("Attemps {}".format(failed_attemps))
            if failed_attemps > 7:
                #return 0, total_cost
                print("Attemps {}".format(failed_attemps))
            pass
    print("Cost was {} dollars".format(total_cost))
    return iPPI_helpfullness, total_cost

def evaluate_node_helpfullness(node, temperature, therapeutic_goal, n_rep_avg, model):
    iP_helpfullness = 0

    total_cost = 0
    reps = 0

    failed_attemps = 0
    while reps < n_rep_avg:
        try:
            gpt_input = "The protein is\n\n\n" + str(get_uniprot_data(node)) + "\n\n\nMake your judgement and only respond with a number from -100 to 100 based on the instructions."
            gpt_prompt = "From -100 to 100. 100 = should inhibit because helps goal a lot and has no side effects, -100 = do not inhibit because does not help goal and has a lot of side effects. How helpful would inibiting the given protein be for achiving the following goal. Make an educated judgement using your comprehensive biological knowledge and systems understanding. The goal is as follows: {}".format(therapeutic_goal)
            gpt_evaluation = gpt_tools.ask_gpt(gpt_input, gpt_prompt, model, temperature)

            print(gpt_evaluation)
            total_cost += get_cost_for_one_eval(model)
        
            iP_helpfullness += float(gpt_evaluation) / 100.0 / float(n_rep_avg)
            reps += 1
        except Exception as e:
            failed_attemps += 1
            print(e)
            print("Attemps {}".format(failed_attemps))
            if failed_attemps > 7:
                #return 0, total_cost
                print("Attemps {}".format(failed_attemps))
            pass
    print("Cost was {} dollars".format(total_cost))
    return iP_helpfullness, total_cost

def evaluate_edges(edge_list, temperature, therapeutic_goal, model, n_rep_avg, interesting_uniprot_ids, search_tree, file_name=None):
    total_cost = 0
    tuples = set()
    for edge in tqdm(edge_list, desc="Evaluating PPI inhibitions", unit="iPPIs"):
        iPPI_helpfullness, cost = evaluate_edge_helpfullness([edge[0], edge[1]], temperature, therapeutic_goal, n_rep_avg, model)
        total_cost += cost
        tuples.add((edge[0], edge[1], edge[2], iPPI_helpfullness))
        print("Total cost is {} dollars".format(total_cost))

    print("Total cost for edge list was {} dollars".format(total_cost))

    json_data = {"iPPI_tuples": list(tuples), "search_tree": search_tree, "interesting_uniprot_ids": interesting_uniprot_ids, "cost": total_cost, "model": model, "n_rep_avg": n_rep_avg, "temperature": temperature, "therapeutic_goal": therapeutic_goal}

    if file_name is not None: # Save, will overwrite.
        with open(file_name, "w") as file:
            json.dump(json_data, file)

    return json_data

def evaluate_nodes(node_list, temperature, therapeutic_goal, model, n_rep_avg, interesting_uniprot_ids, search_tree, file_name=None):
    total_cost = 0
    tuples = set()
    for node in tqdm(node_list, desc="Evaluating protein inhibitions", unit="iPs"):
        iP_helpfullness, cost = evaluate_node_helpfullness(node[0], temperature, therapeutic_goal, n_rep_avg, model)
        total_cost += cost
        tuples.add((node[0], node[1], iP_helpfullness))
        print("Total cost is {} dollars".format(total_cost))

    print("Total cost for edge list was {} dollars".format(total_cost))

    json_data = {"iP_tuples": list(tuples), "search_tree": search_tree, "interesting_uniprot_ids": interesting_uniprot_ids, "cost": total_cost, "model": model, "n_rep_avg": n_rep_avg, "temperature": temperature, "therapeutic_goal": therapeutic_goal}

    if file_name is not None: # Save, will overwrite.
        with open(file_name, "w") as file:
            json.dump(json_data, file)

    return json_data

def both_in_uniprot_list(edge, uniprot_list): # AlphaFold
    if edge[0] in uniprot_list and edge[1] in uniprot_list:
        return True
    return False

def get_only_unique(edge_list):
    unique_pairs = list(set(tuple(sorted(pair)) for pair in edge_list))
    unique_list = [list(pair) for pair in unique_pairs]
    return unique_list

def get_latest_nodes_and_edges_evaluation():
    with open("molytica_m/target_selector/GPT4_nodes.json", "r") as file:
        iP_tuples = json.load(file)["iP_tuples"]
    with open("molytica_m/target_selector/GPT4_edges.json", "r") as file:
        iPPI_tuples = json.load(file)["iPPI_tuples"]
    return iP_tuples, iPPI_tuples

def get_edges_with_PPI_above_threshold(reference_uniprots, threshold):
    af_uniprots = alpha_fold_tools.get_alphafold_uniprot_ids()

    first_neighbors = []

    # Get first step neighbors
    for ref_uniprot in tqdm(reference_uniprots, desc="Searching ref_uniprot links"):
        for af_uniprot in tqdm(af_uniprots, desc="Searching af_uniprot links"):
            PPI_prob = PPI_Interactome_tools.get_ppi_prob(ref_uniprot, af_uniprot)
            if not PPI_prob:
                PPI_prob = PPI_Interactome_tools.predict_PPI_prob_bidirectional(ref_uniprot, af_uniprot)
            if PPI_prob > threshold:
                first_neighbors.append((af_uniprot, PPI_prob))
    
    return first_neighbors
    
def get_top_edges(ref_prots, threshold, edge_limit):
    all_edges = []

    # Step 1: Get all edges with probabilities for ref_prots
    for ref_prot in ref_prots:
        neighbors = get_edges_with_PPI_above_threshold([ref_prot], threshold)
        for neighbor, prob in neighbors:
            all_edges.append((ref_prot, neighbor, prob))

    # Step 2: Sort edges by probability and take top edges
    all_edges.sort(key=lambda x: x[2], reverse=True)
    top_first_step_edges = all_edges[:edge_limit]

    top_second_step_edges = []

    # Step 3: For each top edge, repeat the process for its neighbor
    for _, neighbor, _ in top_first_step_edges:
        second_step_neighbors = get_edges_with_PPI_above_threshold([neighbor], threshold)
        for second_neighbor, prob in second_step_neighbors:
            top_second_step_edges.append((neighbor, second_neighbor, prob))

    # Sorting second step edges and taking the top ones
    top_second_step_edges.sort(key=lambda x: x[2], reverse=True)
    top_second_step_edges = top_second_step_edges[:edge_limit]

    return top_first_step_edges, top_second_step_edges
