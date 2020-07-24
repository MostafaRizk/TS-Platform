from scripts.benchmarking import BenchmarkPlotter
import os

experiments = [("ffnn_bias_0HL", "result_files/ffnn_bias_0HL.csv"),
               ("ffnn_bias_1HL_4HU", "result_files/ffnn_bias_1HL_4HU.csv"),
               ("ffnn_bias_2HL_4HU", "result_files/ffnn_bias_2HL_4HU.csv"),
               ("ffnn_no-bias_0HL", "result_files/ffnn_no-bias_0HL.csv"),
               ("ffnn_no-bias_1HL_4HU", "result_files/ffnn_no-bias_1HL_4HU.csv"),
               ("ffnn_no-bias_2HL_4HU", "result_files/ffnn_no-bias_2HL_4HU.csv"),
               ("rnn_bias_0HL", "result_files/rnn_bias_0HL.csv"),
               ("rnn_bias_1HL_4HU", "result_files/rnn_bias_1HL_4HU.csv"),
               ("rnn_bias_2HL_4HU", "result_files/rnn_bias_2HL_4HU.csv"),
               ("rnn_no-bias_0HL", "result_files/rnn_no-bias_0HL.csv"),
               ("rnn_no-bias_1HL_4HU", "result_files/rnn_no-bias_1HL_4HU.csv"),
               ("rnn_no-bias_2HL_4HU", "result_files/rnn_no-bias_2HL_4HU.csv"),
               ("sliding_0", "result_files/sliding_0.csv"),
               ("sliding_1", "result_files/sliding_1.csv"),
               ("sliding_2", "result_files/sliding_2.csv"),
               ("sliding_3", "result_files/sliding_3.csv"),
               ("sliding_5", "result_files/sliding_5.csv"),
               ("sliding_6", "result_files/sliding_6.csv"),
               ("single_agent", "result_files/single_agent.csv"),
               ("no_battery", "result_files/no_battery.csv"),
               ("sliding_4", "result_files/sliding_4.csv")]

for experiment in experiments:
    experiment_name = experiment[0]
    experiment_filename = experiment[1]

    os.mkdir(f"{experiment_name}")

    plotter = BenchmarkPlotter(f"{experiment_name}/SlopeForaging-{experiment_name}", experiment_filename)
    plotter.save_all_sample_stats()
