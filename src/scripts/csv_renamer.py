import os
from glob import glob


csv_files = glob(f'result_files/all_genomes_*.csv')

for filename in csv_files:
    name_parts = filename.split("_")[-10:-5]

    if name_parts[1] == "False":
        name_parts[1] = "no-bias"
    else:
        name_parts[1] = "bias"

    name_parts[2] += "HL"

    if int(name_parts[3]) > 0:
        name_parts[3] += "HU"
    else:
        del name_parts[3]

    shortened_name = "_".join(name_parts)

    os.rename(filename, f"result_files/{shortened_name}.csv")

'''
# Fix all file and folder names with 0HU in them as it is redundant
redundant_names = glob(f'*0HU_*')

for name in redundant_names:
    os.chdir(name)
    redundant_sub_names = glob(f'*0HU_*')

    for sub_name in redundant_sub_names:
        replacement = sub_name.replace("0HU_", "")
        os.rename(sub_name, replacement)

    os.chdir("../")
    replacement = name.replace("0HU_", "")
    os.rename(name, replacement)
'''