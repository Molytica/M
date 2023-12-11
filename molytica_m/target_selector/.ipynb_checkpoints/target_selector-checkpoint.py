from molytica_m.target_selector import target_selector_tools
import sys

interesting_uniprots = ["Q01860", "Q06416", "P48431", "O43474"] #OSK (ADD SITR1?)
#interesting_uniprots = ["Q01860"]
model = "gpt-4-1106-preview"
search_tree = [800, 80, 20]
temperature = 0.0
n_rep_avg = 1

with open("molytica_m/target_selector/therapeutic_goal.txt", "r") as file:
    therapeutic_goal = file.read()


print(target_selector_tools.monte_carlo_simulate(["Q01860", "Q06416", "P48431", "O43474"]))
print(len(target_selector_tools.monte_carlo_simulate(["O43474"])))