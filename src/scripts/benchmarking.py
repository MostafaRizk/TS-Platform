"""
Repurposes and modifies code from https://github.com/declanoller/rwg-benchmark/ to conduct benchmarking on sampled
genomes
"""

import matplotlib.colors as colours
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import cv2

from datetime import datetime
from glob import glob


class BenchmarkPlotter:
    def __init__(self, env_name, genomes_file):
        self.env_name = env_name
        self.run_dir = env_name
        #os.mkdir(self.run_dir) # Uncomment if creating individual rather than combined plots

        self.dt_str = "" #datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        self.genomes_file = genomes_file

        if self.env_name.split("_")[0] == "slope":
            self.spec_score_keys = ["R_coop", "R_coop_eff", "R_spec", "R_coop x P", "R_coop_eff x P", "R_spec x P"]
        elif self.env_name.split("_")[0] == "tmaze":
            self.spec_score_keys = ["num_pairs"]

        #### Plot params
        self.plot_pt_alpha = 0.2
        self.plot_label_params = {
            'fontsize': 18
        }
        self.plot_tick_params = {
            'fontsize': 13
        }
        self.plot_title_params = {
            'fontsize': 18
        }

    def load_dictionary_from_file(self, num_samples, num_episodes):
        # Best mean score so far
        best_scores = []

        # Mean score for each sample (genome)
        all_scores = []

        # List of scores in all episodes (simulation runs) for each sample (genome)
        # 50,000 lists of length 20, in my case
        all_trials = []

        # Parameters for logging specialisation
        all_spec_scores = {}
        all_spec_trials = {}
        for key in self.spec_score_keys:
            all_spec_scores[key] = []
            all_spec_trials[key] = []

        # Weights of
        best_weights = []

        L0_weights = []
        L1_weights = []
        L2_weights = []

        N_samples = num_samples
        N_episodes = num_episodes
        total_runtime = -1.0

        #####
        f = open(self.genomes_file, "r")
        data = f.read().strip().split("\n")

        num_spec_scores = len(self.spec_score_keys) * N_episodes
        num_scores = N_episodes + num_spec_scores

        best_weights = np.array([float(element) for element in data[0].split(",")[0:-num_scores]])
        best_score = np.mean([float(episode_score) for episode_score in data[0].split(",")[-num_scores:-num_spec_scores]])

        for row in data:
            genome = np.array([float(element) for element in row.split(",")[0:-num_scores]])
            episode_scores = [float(score) for score in row.split(",")[-num_scores:-num_spec_scores]]
            mean_score = np.mean(episode_scores)
            all_scores += [mean_score]

            if mean_score > best_score:
                best_score = mean_score
                best_weights = genome

            best_scores += [best_score]
            all_trials += [episode_scores]

            # Make a list of lists. For each specialisation metric, there is a list of scores for all episodes
            if num_spec_scores == 1:
                raw_spec_scores = [row.split(",")[-1]]
            else:
                raw_spec_scores = row.split(",")[-num_spec_scores:]

            raw_spec_scores = [[float(raw_spec_scores[i + j]) for i in range(0, len(raw_spec_scores), len(self.spec_score_keys))] for j in range(len(self.spec_score_keys))]

            # Take the average of all the episodes for each specialisation metric and store them in all_spec_scores
            # Store the unaveraged scores in all_spec_trials
            for i in range(len(self.spec_score_keys)):
                all_spec_scores[self.spec_score_keys[i]] += [np.mean(raw_spec_scores[i])]
                all_spec_trials[self.spec_score_keys[i]] += [raw_spec_scores[i]]

            L0 = sum([np.sum(w) for w in genome]) / len(genome)
            L1 = sum([np.abs(w).sum() for w in genome]) / len(genome)
            L2 = sum([(w ** 2).sum() for w in genome]) / len(genome)

            L0_weights.append(L0)
            L1_weights.append(L1)
            L2_weights.append(L2)

        #####

        data_dict = {
            'best_scores': best_scores,
            'all_scores': all_scores,
            'all_trials': all_trials,
            'best_weights': [bw.tolist() for bw in best_weights],
            'all_spec_scores': all_spec_scores,
            'all_spec_trials': all_spec_trials,
            'L0_weights': L0_weights,
            'L1_weights': L1_weights,
            'L2_weights': L2_weights,
            'N_samples': N_samples,
            'N_episodes': N_episodes,
            'total_runtime': total_runtime
        }

        return data_dict

    def plot_scores(self, sample_dict, **kwargs):

        '''
        For plotting results. Pass it a dict of the form
        returned by sample().

        Plots several versions of the same data (only mean, in the order they're
        run, mean, but ordered by increasing value, and then the mean and the scores
        for each trial).
        '''

        ###################### In time order

        """
        plt.close('all')
        plt.plot(sample_dict['all_scores'], color='dodgerblue', label='All mean scores')

        plt.xlabel('Sample', **self.plot_label_params)
        plt.ylabel('Sample mean score', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f'{self.env_name}', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_score_mean_timeseries_{}.png'.format(self.env_name, self.dt_str)))
        """

        ###################### In mean order

        """
        all_scores = sample_dict['all_scores']
        all_scores = sorted(all_scores)

        plt.close('all')
        plt.plot(all_scores, color='mediumseagreen')

        plt.xlabel('Sorted by sample mean score', **self.plot_label_params)
        plt.ylabel('Sample mean score', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f'{self.env_name}', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_score_mean_ordered_{}.png'.format(self.env_name, self.dt_str)))
        """

        ###################### In mean order, with all trials

        '''
        all_trials = sample_dict['all_trials']
        all_trials = sorted(all_trials, key=lambda x: np.mean(x))

        all_trials_mean = np.mean(all_trials, axis=1)

        # For coloring by all episode scores
        all_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_trials)]
        all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))

        perc_cutoff = 99.9
        perc_cutoff_val = np.percentile(all_trials_indexed[:, 1], perc_cutoff)

        all_trials_below = np.array([x for x in all_trials_indexed if x[1] < perc_cutoff_val])
        all_trials_above = np.array([x for x in all_trials_indexed if x[1] >= perc_cutoff_val])

        plt.close('all')

        if kwargs.get('mean_lim', None) is not None:
            lims = kwargs.get('mean_lim', None)
            plt.ylim(lims[0], lims[1])

        plt.plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=self.plot_pt_alpha, markersize=3)
        plt.plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=self.plot_pt_alpha, markersize=3)
        plt.plot(all_trials_mean, color='black')

        plt.xlabel('Sorted by $R_a(n)$', **self.plot_label_params)
        plt.ylabel('$S_{a,n,e}$ and $M_{a,n}$', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f'{self.env_name}', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_score_trials_ordered_{}.png'.format(self.env_name, self.dt_str)))
        '''

        ###################### In mean order, with all trials AND specialisation overlayed

        # For coloring by specialisation scores

        spec_metric_key = None

        if kwargs.get('spec_metric_key', None) is None:
            raise RuntimeError("No specialisation metric passed")
        else:
            spec_metric_key = kwargs.get('spec_metric_key', None)

        all_trials = sample_dict['all_trials']
        all_spec_trials = sample_dict['all_spec_trials'][spec_metric_key]

        all_spec_trials = [x for (y, x) in sorted(zip(all_trials, all_spec_trials), key=lambda pair: np.mean(pair[0]))]
        all_trials = sorted(all_trials, key=lambda x: np.mean(x))

        all_trials_mean = np.mean(all_trials, axis=1)

        # [ [   [x1, score], [x1, score], [x1,score]    ],
        #                     ...
        #   [   [xn, score], [xn, score], [xn,score]    ],
        # ]
        all_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_trials)]
        all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))

        all_spec_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_spec_trials)]
        all_spec_trials_indexed = np.array(all_spec_trials_indexed).reshape((-1, 2))

        plt.close('all')

        if kwargs.get('mean_lim', None) is not None:
            lims = kwargs.get('mean_lim', None)
            plt.ylim(lims[0], lims[1])

        cm = plt.cm.get_cmap('RdYlGn')
        norm = colours.Normalize(vmin=0, vmax=1)
        m = plt.cm.ScalarMappable(norm=norm, cmap=cm)

        for score_tuple, spec_tuple in zip(all_trials_indexed, all_spec_trials_indexed):
            colour = m.to_rgba(spec_tuple[1])
            plt.plot(score_tuple[0], score_tuple[1], 'o', color=colour, alpha=self.plot_pt_alpha, markersize=3)

        plt.plot(all_trials_mean, color='black')

        plt.xlabel('Sorted by $R_a(n)$', **self.plot_label_params)
        plt.ylabel('$S_{a,n,e}$ and $M_{a,n}$', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f'{self.env_name}', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_score_trials_ordered_{}.png'.format(self.env_name, spec_metric_key)))

    def plot_all_trial_stats(self, sample_dict, **kwargs):

        '''
        Plots the variance, min, and max of the scores for the N_episodes of
        each episode, as a function of the mean score for that episode.

        '''

        N_samples = len(sample_dict['all_trials'])
        N_episodes = len(sample_dict['all_trials'][0])

        ####################### Episode score variance
        plt.close('all')

        sigma = np.std(sample_dict['all_trials'], axis=1)

        if kwargs.get('var_lim', None) is not None:
            lims = kwargs.get('var_lim', None)
            plt.ylim(lims[0], lims[1])

        if kwargs.get('mean_lim', None) is not None:
            lims = kwargs.get('mean_lim', None)
            plt.xlim(lims[0], lims[1])

        plt.plot(sample_dict['all_scores'], sigma, 'o', color='mediumorchid', alpha=self.plot_pt_alpha)

        plt.xlabel('$M_a(n)$', **self.plot_label_params)
        plt.ylabel('$V_{a,n}$', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{self.env_name}', **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(self.run_dir, '{}_variance_meanscore_{}.png'.format(self.env_name, self.dt_str))
        plt.savefig(fname)

        ####################### Min sample score
        """
        plt.close('all')

        trial_min = np.min(sample_dict['all_trials'], axis=1)

        plt.plot(sample_dict['all_scores'], trial_min, 'o', color='dodgerblue', alpha=self.plot_pt_alpha)

        plt.xlabel('Sample mean score', **self.plot_label_params)
        plt.ylabel('Min of sample scores', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{self.env_name},\n min score of N_episodes = {N_episodes}', **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(self.run_dir, '{}_min_score_{}.png'.format(self.env_name, self.dt_str))
        plt.savefig(fname)
        """

        ####################### Max episode score
        """
        plt.close('all')

        trial_max = np.max(sample_dict['all_trials'], axis=1)

        plt.plot(sample_dict['all_scores'], trial_max, 'o', color='dodgerblue', alpha=self.plot_pt_alpha)

        plt.xlabel('Sample mean score', **self.plot_label_params)
        plt.ylabel('Max of sample scores', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{self.env_name},\n max score of N_episodes = {N_episodes}', **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(self.run_dir, '{}_max_score_{}.png'.format(self.env_name, self.dt_str))
        plt.savefig(fname)
        """

        ####################### Min and max episode score
        """
        plt.close('all')

        trial_min = np.min(sample_dict['all_trials'], axis=1)
        trial_max = np.max(sample_dict['all_trials'], axis=1)

        plt.plot(sample_dict['all_scores'], trial_min, 'o', color='mediumturquoise', alpha=self.plot_pt_alpha)
        plt.plot(sample_dict['all_scores'], trial_max, 'o', color='plum', alpha=self.plot_pt_alpha)

        plt.xlabel('Sample mean score', **self.plot_label_params)
        plt.ylabel('Min and max of sample scores', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.title(f'{self.env_name}, min (turquoise) and \nmax (purple) score of N_episodes = {N_episodes}',
                  **self.plot_title_params)
        plt.tight_layout()
        fname = os.path.join(self.run_dir, '{}_min_max_score_{}.png'.format(self.env_name, self.dt_str))
        plt.savefig(fname)
        """

    def plot_sample_histogram(self, dist, dist_label, fname, **kwargs):

        '''
        For plotting the distribution of various benchmarking stats for self.env_name.
        Plots a vertical dashed line at the mean.

        kwarg plot_log = True also plots one with a log y axis, which is often
        better because the number of best solutions are very small.
        '''

        fname = os.path.join(self.run_dir, fname)

        plt.close('all')
        # mu = np.mean(dist)
        # sd = np.std(dist)

        """
        if kwargs.get('N_bins', None) is None:
            plt.hist(dist, color='dodgerblue', edgecolor='gray')
        else:
            plt.hist(dist, color='dodgerblue', edgecolor='gray', bins=kwargs.get('N_bins', None))

        # plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
        # plt.xlabel(dist_label, **self.plot_label_params)

        plt.xlabel('$M_{a,n}$', **self.plot_label_params)
        plt.ylabel('Counts', **self.plot_label_params)
        # plt.ylabel('$S_{t,n,r}$ and $M_{t,n}$', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.title(f'{dist_label} distribution for {self.env_name}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$', **self.plot_title_params)
        # plt.title(f'{dist_label} distribution \nfor {self.env_name}', **self.plot_title_params)
        plt.title(f'{self.env_name}', **self.plot_title_params)
        plt.savefig(fname)
        """

        if kwargs.get('dist_lim', None) is not None:
            lims = kwargs.get('dist_lim', None)
            plt.ylim(lims[0], lims[1])

        if kwargs.get('plot_log', False):
            if kwargs.get('N_bins', None) is None:
                plt.hist(dist, color='dodgerblue', edgecolor='gray', log=True)
            else:
                plt.hist(dist, color='dodgerblue', edgecolor='gray', bins=kwargs.get('N_bins', None), log=True)

            # plt.axvline(mu, linestyle='dashed', color='tomato', linewidth=2)
            # plt.xlabel(dist_label, **self.plot_label_params)
            plt.xlabel('$M_{a,n}$', **self.plot_label_params)
            plt.ylabel('log(Counts)', **self.plot_label_params)

            plt.xticks(**self.plot_tick_params)
            plt.yticks(**self.plot_tick_params)

            # plt.title(f'{dist_label} distribution for {self.env_name}\n$\mu = {mu:.1f}$, $\sigma = {sd:.1f}$', **self.plot_title_params)
            # plt.title(f'{dist_label} distribution \nfor {self.env_name}', **self.plot_title_params)
            plt.title(f'{self.env_name}', **self.plot_title_params)
            plt.tight_layout()
            plt.savefig(fname.replace('dist', 'log_dist'))

    def plot_weight_stats(self, sample_dict, **kwargs):

        '''
        For plotting episode mean scores and the corresponding L1 or L2 sums
        of the weight matrix that produced those scores.

        '''

        L0_weights = sample_dict['L0_weights']
        L1_weights = sample_dict['L1_weights']
        L2_weights = sample_dict['L2_weights']
        all_scores = sample_dict['all_scores']

        ###################### L0
        plt.close('all')
        plt.plot(all_scores, L0_weights, 'o', color='forestgreen', alpha=self.plot_pt_alpha)

        plt.xlabel('Sample mean score', **self.plot_label_params)
        plt.ylabel('L0/N_weights', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f'{self.env_name},\n L0 sum of weights', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_L0_vs_meanscore_{}.png'.format(self.env_name, self.dt_str)))

        ###################### L1
        plt.close('all')
        plt.plot(all_scores, L1_weights, 'o', color='forestgreen', alpha=self.plot_pt_alpha)

        plt.xlabel('Sample mean score', **self.plot_label_params)
        plt.ylabel('L1/N_weights', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f'{self.env_name},\n L1 sum of weights', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_L1_vs_meanscore_{}.png'.format(self.env_name, self.dt_str)))

        ######################## L2
        plt.close('all')
        plt.plot(all_scores, L2_weights, 'o', color='forestgreen', alpha=self.plot_pt_alpha)

        plt.xlabel('Sample mean score', **self.plot_label_params)
        plt.ylabel('L2/N_weights', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        # plt.legend()
        plt.title(f'{self.env_name},\n L2 sum of weights', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_L2_vs_meanscore_{}.png'.format(self.env_name, self.dt_str)))

    def plot_score_percentiles(self, sample_dict, **kwargs):

        ###################### In mean order, with all trials

        all_trials = sample_dict['all_trials']
        all_trials_mean = np.mean(all_trials, axis=1)

        percs = np.linspace(0, 100.0, 20)
        perc_values = np.percentile(all_trials_mean, percs)

        percs_10 = np.arange(10, 100, 5).tolist() + [96, 97, 98, 99]
        perc_10_values = np.percentile(all_trials_mean, percs_10)

        plt.close('all')
        # plt.plot(all_trials, 'o', color='tomato', alpha=self.plot_pt_alpha)
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        for perc, val in zip(percs_10, perc_10_values):
            ax.axvline(perc, linestyle='dashed', color='gray', linewidth=0.5, label=f'perc {perc} = {val:.1f}')
            ax.axhline(val, linestyle='dashed', color='gray', linewidth=0.5)

        plt.plot(percs, perc_values, color='mediumseagreen')

        plt.xlabel('Percentile', **self.plot_label_params)
        plt.ylabel('Percentile value', **self.plot_label_params)

        plt.xticks(**self.plot_tick_params)
        plt.yticks(**self.plot_tick_params)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'{self.env_name}', **self.plot_title_params)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, '{}_percentiles_{}.png'.format(self.env_name, self.dt_str)))

    def save_sample_dict(self, sample_dict):
        '''
        For saving the results of the run in a .json file, for later analysis.
        '''

        # Maybe not necessary, but to be careful to not modify the original
        sample_dict_copy = sample_dict.copy()
        if 'best_weights' in sample_dict_copy.keys():
            sample_dict_copy.pop('best_weights')

        fname = f'{self.genomes_file.strip(".csv")}stats.json'
        # Save distributions to file

        with open(fname, 'w+') as f:
            json.dump(sample_dict_copy, f, indent=4)

    def save_all_sample_stats(self, **kwargs):
        """
        For saving all the stats and plots for the sampling, just a collector
        function.

        """
        if kwargs.get('num_samples', None) is None:
            raise RuntimeError("Number of samples not passed")
        else:
            num_samples = kwargs.get('num_samples', None)

        if kwargs.get('num_episodes', None) is None:
            raise RuntimeError("Number of episodes not passed")
        else:
            num_episodes = kwargs.get('num_episodes', None)

        sample_dict = self.load_dictionary_from_file(num_samples=num_samples, num_episodes=num_episodes)
        self.save_sample_dict(sample_dict)

        if kwargs.get('save_plots', True):
            for key in self.spec_score_keys[:3]:
                self.plot_scores(sample_dict, spec_metric_key=key, **kwargs)

            self.plot_all_trial_stats(sample_dict, **kwargs)
            self.plot_sample_histogram(sample_dict['all_scores'], 'Mean sample score', f'{self.env_name}_all_scores_dist_{self.dt_str}.png', plot_log=True, **kwargs)
            #self.plot_weight_stats(sample_dict)
            #self.plot_score_percentiles(sample_dict)

def get_experiment_name_from_filename(filename):
    env_name = filename.strip(".csv").split("/")[-1].split("_")[7]
    learning_type = filename.strip(".csv").split("/")[-1].split("_")[2]
    items_in_shortened_name = filename.strip(".csv").split("_")[-10:-6]
    shortened_name = "_".join([str(item) for item in items_in_shortened_name])
    return "_".join([env_name, learning_type, shortened_name])

def arch_dict_to_label(arch_dict):
    label = '{} HL'.format(arch_dict['N_hidden_layers'])
    if 'N_hidden_units' in arch_dict.keys():
        label += ', {} HU'.format(arch_dict['N_hidden_units'])

    return label

def walk_multi_dir(results_dir, bias, params_dict_list, **kwargs):
    if kwargs.get('spec_metric_key', None) is None:
        raise RuntimeError("No specialisation metric passed")
    else:
        spec_metric_key = kwargs.get('spec_metric_key', None)

    if kwargs.get('env', None) is None:
        raise RuntimeError("No env name passed")
    else:
        env = kwargs.get('env', None)

    params_results_dict_list = []
    for params_dict in params_dict_list:

        stat_file_prefix = f'all_genomes_centralised_rwg_heterogeneous_team_nn_{env}*{params_dict["NN"].lower()}_{bias}_{params_dict["N_hidden_layers"]}_{params_dict["N_hidden_units"]}'
        regex_string = f'{results_dir}/{stat_file_prefix}_*_stats.json'
        stat_files = glob(regex_string)

        if len(stat_files) > 1:
            raise RuntimeError('Too many json files with same parameters')

        elif len(stat_files) == 1:
            with open(stat_files[0], 'r') as f:
                sample_stats = json.load(f)

        elif len(stat_files) < 1:
            regex_string = f'{results_dir}/{stat_file_prefix}*csv'
            csv_files = glob(regex_string)

            if len(csv_files) > 1:
                raise RuntimeError(f"Too many csv files with parameters {regex_string}")
            elif len(csv_files) < 1:
                raise RuntimeError(f"No csv files with parameters {regex_string}")

            genomes_file = csv_files[0]  # os.path.join(results_dir, csv_files[0])
            experiment_name = get_experiment_name_from_filename(genomes_file)

            if kwargs.get('num_samples', None) is None:
                raise RuntimeError("Number of samples not passed")
            else:
                num_samples = kwargs.get('num_samples', None)

            if kwargs.get('num_episodes', None) is None:
                raise RuntimeError("Number of episodes not passed")
            else:
                num_episodes = kwargs.get('num_episodes', None)

            plotter = BenchmarkPlotter(experiment_name, genomes_file)
            sample_stats = plotter.load_dictionary_from_file(num_samples=num_samples, num_episodes=num_episodes)
            plotter.save_sample_dict(sample_stats)

        all_trials = np.array(sample_stats['all_trials'])
        all_trials_mean = np.mean(all_trials, axis=1)
        all_spec_trials = sample_stats['all_spec_trials'][spec_metric_key]
        params_results = params_dict.copy()
        params_results['best_score'] = np.max(all_trials_mean)
        params_results['percentile_99.9'] = np.percentile(all_trials_mean, 99.9)
        params_results['all_trials'] = all_trials
        params_results['all_spec_trials'] = all_spec_trials
        if 'total_runtime' in sample_stats.keys():
            params_results['total_runtime'] = sample_stats['total_runtime']
        params_results_dict_list.append(params_results)

    # pp.pprint(params_results_dict_list)
    return params_results_dict_list

def plot_envs_vs_NN_arch(parent_dir, bias, **kwargs):
    '''
    For plotting a nx5 grid of envs on one axis, and NN arch's used to
    solve them on the other axis.
    '''

    results_dir = os.path.join(parent_dir, 'data')
    analysis_dir = os.path.join(parent_dir, 'analysis')
    figures_dir = os.path.join(analysis_dir, f'combined_plots_{bias}')

    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    if kwargs.get('spec_metric_key', None) is None:
        raise RuntimeError("No specialisation metric passed")
    else:
        spec_metric_key = kwargs.get('spec_metric_key', None)

    if kwargs.get('env', None) is None:
        raise RuntimeError("No env name passed")
    else:
        env = kwargs.get('env', None)

    print(f'Making plots for {spec_metric_key}')

    if env == "slope":
        envs_list = [
            'SlopeForaging-FFNN',
            'SlopeForaging-RNN'
        ]

        env_name_title_dict = {
            'SlopeForaging-FFNN': f'SlopeForaging\n(FFNN)',
            'SlopeForaging-RNN': f'SlopeForaging\n(RNN)'
        }

    elif env == "tmaze":
        envs_list = ["TMaze-FFNN"]
        env_name_title_dict = {
            'TMaze-FFNN': f'TMaze\n(FFNN)'
        }

    '''{
        'N_hidden_layers' : 1,
        'N_hidden_units' : 2
    },
    '''
    arch_dict_list = [
        {
            'N_hidden_layers': 0,
            'N_hidden_units': 0
        },

        {
            'N_hidden_layers': 1,
            'N_hidden_units': 4
        },
        {
            'N_hidden_layers': 2,
            'N_hidden_units': 4
        },
    ]

    params_dict_list = []

    for env in envs_list:
        for arch_dict in arch_dict_list:
            params_dict = arch_dict.copy()
            params_dict['env_name'] = env
            params_dict['NN'] = env.split('-')[1]
            params_dict_list.append(params_dict)

    walk_dict_list = walk_multi_dir(results_dir, bias, params_dict_list, **kwargs)

    env_arch_score_dict = {}
    env_arch_spec_score_dict = {}
    for w_dict in walk_dict_list:

        env_arch_list = [w_dict['env_name'], w_dict['N_hidden_layers']]
        if 'N_hidden_units' in w_dict.keys():
            env_arch_list.append(w_dict['N_hidden_units'])

        env_arch_tuple = tuple(env_arch_list)
        print(env_arch_tuple)
        env_arch_score_dict[env_arch_tuple] = w_dict['all_trials']
        env_arch_spec_score_dict[env_arch_tuple] = w_dict['all_spec_trials']

    unit_plot_w = 2.7
    unit_plot_h = 0.55 * unit_plot_w

    grid_fig_size = (unit_plot_w * len(envs_list), unit_plot_h * len(arch_dict_list))
    plot_pt_alpha = 0.1
    plot_label_params = {
        'fontsize': 13
    }
    plot_ylabel_params = {
        'fontsize': 12
    }
    plot_tick_params = {
        'axis': 'both',
        'labelsize': 16
    }
    plot_title_params = {
        'fontsize': 12
    }

    print('\nPlotting big combo plot...')

    w_space = 0
    fig_height = 8
    adjust_kwargs = {
        'bottom': 0.12,
        'right': 0.98,
        'top': 0.99
    }
    part_1_w = fig_height / 1.0
    part_2_w = (3 / 4.0) * part_1_w * 1.0
    plot_tick_params = {
        'axis': 'both',
        'labelsize': 6,
        'which': 'major',
        'pad': 0

    }

    ############################# First part
    plt.close('all')
    N_row = len(envs_list)
    N_col = len(arch_dict_list)
    fig, axes = plt.subplots(N_row, N_col, sharex='all', sharey='row',
                             gridspec_kw={'hspace': 0, 'wspace': w_space}, figsize=(part_1_w, fig_height))

    ###################### plot main episode/mean plots
    for i, env_name in enumerate(envs_list):

        if len(envs_list) == 1:
            axes_list = axes
        else:
            axes_list = axes[i]

        for j, arch_dict in enumerate(arch_dict_list):

            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]
            spec_scores = env_arch_spec_score_dict[env_arch_tuple][:10000]

            all_spec_trials = [x for (y, x) in sorted(zip(scores, spec_scores), key=lambda pair: np.mean(pair[0]))]
            all_trials = sorted(scores, key=lambda x: np.mean(x))

            all_trials_mean = np.mean(all_trials, axis=1)

            # For coloring by all episode scores
            all_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_trials)]
            all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))

            all_spec_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_spec_trials)]
            all_spec_trials_indexed = np.array(all_spec_trials_indexed).reshape((-1, 2))

            if kwargs.get('mean_lim', None) is None:
                raise RuntimeError("Did not specify y-limits for mean plot")

            lims = kwargs.get('mean_lim', None)
            axes_list[j].set_ylim(lims[0], lims[1])

            cm = plt.cm.get_cmap('RdYlGn')
            norm = colours.Normalize(vmin=0, vmax=1)
            m = plt.cm.ScalarMappable(norm=norm, cmap=cm)

            for score_tuple, spec_tuple in zip(all_trials_indexed, all_spec_trials_indexed):
                colour = m.to_rgba(spec_tuple[1])
                axes_list[j].plot(score_tuple[0], score_tuple[1], 'o', color=colour, alpha=plot_pt_alpha, markersize=3)

            axes_list[j].plot(all_trials_mean, color='black')

            axes_list[j].tick_params(**plot_tick_params)

            # arch_dict_to_label(arch_dict)
            if i == len(envs_list) - 1:
                axes_list[j].set_xlabel('$R_a(n)$,\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)
            if j == 0:
                axes_list[j].set_ylabel(env_name_title_dict[env_name] + '\n\n$S_{a,n,e}$ and $M_{a,n}$', **plot_label_params)

    plt.subplots_adjust(left=0.2, **adjust_kwargs)
    print('Plotting part 1 png...')
    part_1_path = os.path.join(figures_dir, f'combo_part1_{spec_metric_key}.png')
    plt.savefig(part_1_path, dpi=300)

    ################################ second part
    plt.close('all')
    N_row = len(envs_list)
    N_col = 2
    fig, axes = plt.subplots(N_row, N_col, sharex=False, sharey=False, gridspec_kw={'hspace': 0, 'wspace': 0.23}, figsize=(part_2_w, fig_height))

    ######################### Plot variance for 2HL4HU
    j = 0
    for i, env_name in enumerate(envs_list):
        arch_dict = {
            'N_hidden_layers': 2,
            'N_hidden_units': 4
        }
        env_arch_tuple = (env_name, *list(arch_dict.values()))
        print(f'Plotting mean and trials of {env_arch_tuple}...')
        scores = env_arch_score_dict[env_arch_tuple][:10000]

        all_trials = sorted(scores, key=lambda x: np.mean(x))
        all_trials_mean = np.mean(all_trials, axis=1)
        all_trials_std = np.std(all_trials, axis=1)

        if kwargs.get('var_lim', None) is None:
            raise RuntimeError("Did not specify y-limits for variance plot")

        lims = kwargs.get('var_lim', None)
        axes_list[j].set_ylim(lims[0], lims[1])

        if kwargs.get('mean_lim', None) is None:
            raise RuntimeError("Did not specify x-limits for variance plot")

        lims = kwargs.get('mean_lim', None)
        axes_list[j].set_xlim(lims[0], lims[1])

        axes_list[j].tick_params(**plot_tick_params)
        axes_list[j].tick_params(axis='x', labelsize=0)

        axes_list[j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha,
                        markersize=3)

        # axes_list[j].set_xlabel('$M_{a,n}$', **plot_label_params)
        axes_list[j].set_ylabel('$V_{a,n}$', **plot_label_params)
        if i == len(envs_list) - 1:
            axes_list[j].set_xlabel('$M_{a,n}$\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)

        # axes_list[j].label_outer()

    ##################################### Plot histograms for 2HL4HU
    j = 1
    for i, env_name in enumerate(envs_list):
        arch_dict = {
            'N_hidden_layers': 2,
            'N_hidden_units': 4
        }
        env_arch_tuple = (env_name, *list(arch_dict.values()))
        print(f'Plotting mean and trials of {env_arch_tuple}...')
        scores = env_arch_score_dict[env_arch_tuple][:10000]

        all_trials = sorted(scores, key=lambda x: np.mean(x))
        all_trials_mean = np.mean(all_trials, axis=1)

        if kwargs.get('dist_lim', None) is None:
            raise RuntimeError("Did not specify y-limits for histogram")

        lims = kwargs.get('dist_lim', None)
        axes_list[j].set_ylim(lims[0], lims[1])

        if kwargs.get('N_bins', None) is None:
            raise RuntimeError("Did not specify bins for histogram")

        axes_list[j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True,
                        bins=kwargs.get('N_bins', None))
        axes_list[j].tick_params(**plot_tick_params)
        axes_list[j].tick_params(axis='x', labelsize=0)

        axes_list[j].set_ylabel('', **plot_label_params)
        if i == len(envs_list) - 1:
            axes_list[j].set_xlabel('$M_{a,n}$\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)

        # axes_list[j].label_outer()

    # plt.tight_layout()
    plt.subplots_adjust(left=0.2, **adjust_kwargs)
    print('Plotting part 2 png...')
    part_2_path = os.path.join(figures_dir, f'combo_part2_{spec_metric_key}.png')
    plt.savefig(part_2_path, dpi=300)

    # Save combined plots
    im1 = cv2.imread(part_1_path)
    im2 = cv2.imread(part_2_path)
    combined_path = os.path.join(figures_dir, f'combo_full_{spec_metric_key}.png')
    combined_plot = cv2.hconcat([im1, im2])
    cv2.imwrite(combined_path, combined_plot)
