source_file = "/Users/mostafa/Documents/Code/PhD/TS-Platform/src/scripts/data/spec_plots_debugging/rwg_with_spec.csv"
destination_file = "/Users/mostafa/Documents/Code/PhD/TS-Platform/src/scripts/data/spec_plots_debugging/rwg_with_spec_subset.csv"

f = open(source_file, "r")
raw_data = f.read().strip().split("\n")
f.close()

subset = raw_data[0:100]
g = open(destination_file, "w")
g.write("\n".join([str(entry) for entry in subset]))
g.close()