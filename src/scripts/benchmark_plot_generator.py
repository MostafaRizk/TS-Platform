from scripts.benchmarking import BenchmarkPlotter
import os

experiments = [("ffnn_bias_0HL", "ffnn_bias_0HL.csv"),
               ("ffnn_bias_1HL_4HU", "ffnn_bias_1HL_4HU.csv"),
               ("ffnn_bias_2HL_4HU", "ffnn_bias_2HL_4HU.csv"),
               ("ffnn_no-bias_0HL", "ffnn_no-bias_0HL.csv"),
               ("ffnn_no-bias_1HL_4HU", "ffnn_no-bias_1HL_4HU.csv"),
               ("ffnn_no-bias_2HL_4HU", "ffnn_no-bias_2HL_4HU.csv"),
               ("rnn_bias_0HL", "rnn_bias_0HL.csv"),
               ("rnn_bias_1HL_4HU", "rnn_bias_1HL_4HU.csv"),
               ("rnn_bias_2HL_4HU", "rnn_bias_2HL_4HU.csv"),
               ("rnn_no-bias_0HL", "rnn_no-bias_0HL.csv"),
               ("rnn_no-bias_1HL_4HU", "rnn_no-bias_1HL_4HU.csv"),
               ("rnn_no-bias_2HL_4HU", "rnn_no-bias_2HL_4HU.csv"),
               ("sliding_0", "sliding_0.csv"),
               ("sliding_1", "sliding_1.csv"),
               ("sliding_2", "sliding_2.csv"),
               ("sliding_3", "sliding_3.csv"),
               ("sliding_5", "sliding_5.csv"),
               ("sliding_6", "sliding_6.csv"),
               ("single_agent", "single_agent.csv"),
               ("no_battery", "no_battery.csv"),
               ("sliding_4", "sliding_4.csv")]

for experiment in experiments:
    experiment_name = experiment[0]
    experiment_filename = experiment[1]

    os.mkdir(f"{experiment_name}")

    plotter = BenchmarkPlotter(f"{experiment_name}/SlopeForaging-{experiment_name}", experiment_filename)
    plotter.save_all_sample_stats()
