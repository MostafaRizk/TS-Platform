probs = "0.99900	0.00024	0.00013	0.00063 \
0.00000	0.99900	0.00041	0.00059 \
0.00000	0.00063	0.99900	0.00037 \
0.00000	0.00071	0.00029	0.99900"

strategies = ["novice", "generalist", "dropper", "collector"]

probs = probs.split()
count=0

new_dict = {"mutation_probability": {}}
for strat1 in strategies:
    new_dict["mutation_probability"][strat1] = {}
    for strat2 in strategies:
        new_dict["mutation_probability"][strat1][strat2] = float(probs[count])
        count += 1

new_dict = str(new_dict).replace("\'", "\"").replace("}", "}\n")[1:-4] + "},"
print(new_dict)

