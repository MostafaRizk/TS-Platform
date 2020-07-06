library(flacco)
library(readr)

data <- read_csv("Documents/Code/PhD/TS-Platform/src/all_genomes_rwg_homogeneous_team_nn_1_2_3_1_4_8_4_1_3_7_1_3.0_0.2_2_1000_500_5_0_linear_100000_normal_0_1_.csv", col_names = FALSE)

X = data[,c(1:288)]
y = t(data[,c(289)])

X_mini = X[c(1:10000),]
y_mini = y[,c(1:10000)]

feat.object = createFeatureObject(X = X_mini, y = y_mini)
calculateFeatureSet(feat.object, set = "ela_distr")