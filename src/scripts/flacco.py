from pflacco.pflacco import create_initial_sample, create_feature_object, calculate_feature_set, calculate_features
import numpy as np

# Load samples into x (list of genomes/observations) and y (list of fitnesses/objective values)
data_file = "../all_genomes_rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_100000_normal_0_1_.csv"
f = open(data_file, "r")
lines = f.read().strip().split("\n")
x = np.array([np.array([float(element) for element in row.split(",")[0:-1]]) for row in lines])
y = np.array([float(row.split(",")[-1]) for row in lines])
x_mini = x[0:10001, :]
y_mini = y[0:10001]

# Create feature object. minimize=False because the objective is to maximise. Lower and upper are arbitrarily chosen to
# be outside the range of the sampled genomes
feat_object = create_feature_object(x=x_mini, y=y_mini, minimize=False, lower=-10, upper=10)
y_dist_features = calculate_feature_set(feat_object, 'ela_distr')
print(y_dist_features)

