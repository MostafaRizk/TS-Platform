import os
from glob import glob

csv_files = glob(f'result_files_het/all_genomes_*.csv')

for filename in csv_files:
    name_parts = filename.split("_")[-10:-5]

    if name_parts[1] == "False":
        name_parts[1] = "no-bias"
    else:
        name_parts[1] = "bias"

    name_parts[2] += "HL"
    name_parts[3] += "HU"

    shortened_name = "_".join(name_parts)

    os.rename(filename, f"result_files_het/{shortened_name}.csv")