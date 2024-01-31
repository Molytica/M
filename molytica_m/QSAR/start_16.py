import os

n = 12
commands = []
for x in range(n):
    range_ids = 1136045 - 0
    start_id = 0 + x * (range_ids // n)

    command = f"conda activate M && python molytica_m/QSAR/start_id.py {start_id}"

    commands.append(command)

with open("molytica_m/QSAR/commands.txt", "w") as f:
    f.write("\n".join(commands))