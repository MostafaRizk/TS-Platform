"""
Repurposes and modifies code from https://github.com/declanoller/rwg-benchmark/ to conduct benchmarking on sampled
genomes
"""

import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import json
from glob import glob
import cv2


class BenchmarkPlotter:
    def __init__(self, env_name, genomes_file):
        self.env_name = env_name
        self.run_dir = env_name
        os.mkdir(self.run_dir)

        self.dt_str = ""  # datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        self.genomes_file = genomes_file

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
        # 10,000 lists of length 20, in my case
        all_trials = []

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

        best_weights = np.array([float(element) for element in data[0].split(",")[0:-N_episodes]])
        best_score = np.mean([float(episode_score) for episode_score in data[0].split(",")[-N_episodes:]])

        for row in data:
            genome = np.array([float(element) for element in row.split(",")[0:-N_episodes]])
            episode_scores = [float(score) for score in row.split(",")[-N_episodes:]]
            mean_score = np.mean(episode_scores)
            all_scores += [mean_score]

            if mean_score > best_score:
                best_score = mean_score
                best_weights = genome

            best_scores += [best_score]
            all_trials += [episode_scores]

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

        fname = os.path.join(self.run_dir, 'sample_stats.json')
        # Save distributions to file

        with open(fname, 'w+') as f:
            json.dump(sample_dict_copy, f, indent=4)

    def save_all_sample_stats(self, **kwargs):
        """
        For saving all the stats and plots for the sampling, just a collector
        function.

        """

        sample_dict = self.load_dictionary_from_file(num_samples=10000, num_episodes=20)
        self.save_sample_dict(sample_dict)

        '''
        if kwargs.get('save_plots', True):
            self.plot_scores(sample_dict, **kwargs)
            self.plot_all_trial_stats(sample_dict, **kwargs)
            self.plot_sample_histogram(sample_dict['all_scores'], 'Mean sample score',
                                       f'{self.env_name}_all_scores_dist_{self.dt_str}.png', plot_log=True, **kwargs)
            # self.plot_weight_stats(sample_dict)
            # self.plot_score_percentiles(sample_dict)
        '''

def arch_dict_to_label(arch_dict):
    label = '{} HL'.format(arch_dict['N_hidden_layers'])
    if 'N_hidden_units' in arch_dict.keys():
        label += ', {} HU'.format(arch_dict['N_hidden_units'])

    return label

def walk_multi_dir(stats_dir, bias, params_dict_list):

    params_results_dict_list = []
    for params_dict in params_dict_list:

        regex_string = f'{stats_dir}/all_genomes_rwg_heterogeneous_team_nn_slope*{params_dict["NN"].lower()}_{bias}_{params_dict["N_hidden_layers"]}_{params_dict["N_hidden_units"]}_*'
        possible_folders = glob(regex_string)

        if len(possible_folders) > 1:
            raise RuntimeError('Too many folders with same parameters')
        if len(possible_folders) < 1:
            raise RuntimeError('No folders with same parameters')

        # Get the sample_stats.
        with open(os.path.join(possible_folders[0], 'sample_stats.json'), 'r') as f:
                sample_stats = json.load(f)

        all_trials = np.array(sample_stats['all_trials'])
        all_trials_mean = np.mean(all_trials, axis=1)
        params_results = params_dict.copy()
        params_results['best_score'] = np.max(all_trials_mean)
        params_results['percentile_99.9'] = np.percentile(all_trials_mean, 99.9)
        params_results['all_trials'] = all_trials
        if 'total_runtime' in sample_stats.keys():
            params_results['total_runtime'] = sample_stats['total_runtime']
        params_results_dict_list.append(params_results)

    # pp.pprint(params_results_dict_list)
    return params_results_dict_list

def plot_envs_vs_NN_arch(stats_dir, bias, **kwargs):
    '''
    For plotting a nx5 grid of envs on one axis, and NN arch's used to
    solve them on the other axis.
    '''

    figures_dir = os.path.join(stats_dir, f'combined_plots_{bias}')

    if not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    '''all_runs_dir = os.path.join(stats_dir, 'all_runs')
    # Load csv that holds the names of all the dirs
    stats_overview_fname = os.path.join(stats_dir, 'vary_params_stats.csv')
    overview_df = pd.read_csv(stats_overview_fname)
    drop_list = ['result_ID', 'run_plot_label', 'mu_all_scores', 'sigma_all_scores', 'all_scores_99_perc']
    overview_df = overview_df.drop(drop_list, axis=1)'''

    envs_list = [
        'SlopeForaging-FFNN',
        'SlopeForaging-RNN'
    ]

    env_name_title_dict = {
        'SlopeForaging-FFNN': 'SlopeForaging\n(FFNN)',
        'SlopeForaging-RNN': 'SlopeForaging\n(RNN)'
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

    walk_dict_list = walk_multi_dir(stats_dir, bias, params_dict_list)

    env_arch_score_dict = {}
    for w_dict in walk_dict_list:

        env_arch_list = [w_dict['env_name'], w_dict['N_hidden_layers']]
        if 'N_hidden_units' in w_dict.keys():
            env_arch_list.append(w_dict['N_hidden_units'])

        env_arch_tuple = tuple(env_arch_list)
        print(env_arch_tuple)
        env_arch_score_dict[env_arch_tuple] = w_dict['all_trials']

    '''for i, env_name in enumerate(envs_list):
        for j, arch_dict in enumerate(arch_list):

            subset_df = overview_df[
                (overview_df['env_name'] == env_name) & \
                (overview_df['N_hidden_layers'] == arch_dict['N_hidden_layers']) & \
                (overview_df['N_hidden_units'] == arch_dict['N_hidden_units'])
                ]

            run_fname_label = subset_df['run_fname_label'].values[0]

            match_dirs = [x for x in os.listdir(all_runs_dir) if run_fname_label in x]
            assert len(match_dirs)==1, 'Must only have one dir matching label!'
            vary_dir = match_dirs[0]
            print(vary_dir)

            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(env_arch_tuple)

            for root, dirs, files in os.walk(os.path.join(all_runs_dir, vary_dir)):
                if 'sample_stats.json' in files:
                    with open(os.path.join(root, 'sample_stats.json'), 'r') as f:
                        sample_dict = json.load(f)

                    env_arch_score_dict[env_arch_tuple] = sample_dict['all_trials']'''

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

    plot_score_trials = False
    plot_variances = False
    plot_hists = False
    plot_combo_plot_v1 = False
    plot_combo_plot_v2 = False
    plot_combo_plot_v3 = True

    ####################################### Score trials
    if plot_score_trials:
        print('\nPlotting all score trials...')
        plt.close('all')
        '''fig, axes = plt.subplots(len(envs_list), len(arch_dict_list), sharex='col', sharey='row',
                                    gridspec_kw={'hspace': .1, 'wspace': 0}, figsize=(10,8))'''
        fig, axes = plt.subplots(len(arch_dict_list), len(envs_list), sharex='col', sharey='col',
                                 gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=grid_fig_size)

        for j, env_name in enumerate(envs_list):
            for i, arch_dict in enumerate(arch_dict_list):
                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting mean and trials of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)

                # For coloring by all episode scores
                all_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_trials)]
                all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
                perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_indexed[:, 1], perc_cutoff)

                all_trials_below = np.array([x for x in all_trials_indexed if x[1] < perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials_indexed if x[1] >= perc_cutoff_val])
                axes[i][j].plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=plot_pt_alpha,
                                markersize=3)
                axes[i][j].plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=plot_pt_alpha,
                                markersize=3)

                # For uniform coloring
                # axes[i][j].plot(range(len(all_trials)), all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)

                # For coloring by mean score
                '''perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_mean, perc_cutoff)
                all_trials_below = np.array([x for x in all_trials if np.mean(x)<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials if np.mean(x)>=perc_cutoff_val])
                axes[i][j].plot(range(len(all_trials_below)), all_trials_below, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(range(len(all_trials_mean)-len(all_trials_above), len(all_trials_mean)), all_trials_above, 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)
                '''

                axes[i][j].plot(all_trials_mean, color='black')

                axes[i][j].tick_params(**plot_tick_params)
                '''axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
                axes[i][j].set_ylabel(env_name_title_dict[env_name], **plot_label_params)'''
                axes[i][j].set_ylabel(arch_dict_to_label(arch_dict), **plot_label_params)
                axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                axes[i][j].label_outer()

        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.97)
        print('Plotting png...')
        plt.savefig(os.path.join(figures_dir, f'combo_trials_sorted.png'), dpi=300)
        # print('Plotting pdf...')
        # plt.savefig(os.path.join(figures_dir, f'5x5_trials_sorted.pdf'), dpi=300)

    ####################################### Variances
    if plot_variances:
        print('\nPlotting all variances...')
        plt.close('all')
        fig, axes = plt.subplots(len(envs_list), len(arch_dict_list), sharex=False, sharey='row',
                                 gridspec_kw={'hspace': .5, 'wspace': 0}, figsize=grid_fig_size)

        for i, env_name in enumerate(envs_list):
            for j, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting variance of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]
                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)
                all_trials_std = np.std(all_trials, axis=1)

                axes[i][j].tick_params(**plot_tick_params)

                axes[i][j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha,
                                markersize=3)

                if i == len(envs_list) - 1:
                    axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
                if j == 0:
                    axes[i][j].set_ylabel(env_name_title_dict[env_name], **plot_label_params)
                # axes[i][j].label_outer()

        plt.savefig(os.path.join(figures_dir, f'combo_variance.png'))

    ####################################### Histograms
    if plot_hists:

        print('\nPlotting all histograms...')
        plt.close('all')
        fig, axes = plt.subplots(len(arch_dict_list), len(envs_list), sharex=False, sharey=False,
                                 gridspec_kw={'hspace': 0.25, 'wspace': 0.25}, figsize=grid_fig_size)

        '''for i, env_name in enumerate(envs_list):
        for j, arch_dict in enumerate(arch_dict_list):'''
        for j, env_name in enumerate(envs_list):
            for i, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting log hist of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)

                axes[i][j].tick_params(**plot_tick_params, pad=0)
                # axes[i][j].tick_params(labelsize=8)

                axes[i][j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True, bins=20)
                if i == len(arch_dict_list) - 1:
                    axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                else:
                    axes[i][j].tick_params(axis='x', labelsize=0)

                if env_name == 'MountainCarContinuous-v0':
                    axes[i][j].set_xlim((-60, 110))
                if env_name == 'MountainCar-v0':
                    axes[i][j].set_xlim((-205, -120))
                if env_name == 'Pendulum-v0':
                    axes[i][j].set_xlim((-2000, -750))
                if env_name == 'Acrobot-v1':
                    axes[i][j].set_xlim((-500, -50))
                if env_name == 'SlopeForaging-FFNN' or env_name == 'SlopeForaging-RNN':
                    axes[i][j].set_xlim((-20000, 200000))

                # if i == len(envs_list)-1:
                if j == 0:
                    # axes[i][j].set_ylabel('No bias,\n 2HL 4HU', **plot_label_params)
                    axes[i][j].set_ylabel(kwargs.get('ylabel_prefix', '') + arch_dict_to_label(arch_dict),
                                          **plot_ylabel_params)
                # axes[i][j].label_outer()

        plt.subplots_adjust(bottom=0.18, top=0.98, left=0.08, right=0.99)
        plt.savefig(os.path.join(figures_dir, f'combo_hist_log.png'))

    ####################################### combo plot v1
    if plot_combo_plot_v1:
        print('\nPlotting big combo plot...')
        plt.close('all')
        '''fig, axes = plt.subplots(len(envs_list), len(arch_dict_list), sharex='col', sharey='row',
                                    gridspec_kw={'hspace': .1, 'wspace': 0}, figsize=(10,8))'''
        N_row = len(arch_dict_list) + 2
        N_col = len(envs_list)
        fig, axes = plt.subplots(N_row, N_col, sharex=False, sharey=False,
                                 gridspec_kw={'hspace': .25, 'wspace': 0.3}, figsize=(10, 8))

        plot_tick_params = {
            'axis': 'both',
            'labelsize': 7,
            'which': 'major',
            'pad': 0

        }

        ############################# plot main episode/mean plots
        for j, env_name in enumerate(envs_list):
            for i, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting mean and trials of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)

                # For coloring by all episode scores
                all_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_trials)]
                all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
                perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_indexed[:, 1], perc_cutoff)

                all_trials_below = np.array([x for x in all_trials_indexed if x[1] < perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials_indexed if x[1] >= perc_cutoff_val])
                axes[i][j].plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=plot_pt_alpha,
                                markersize=3)
                axes[i][j].plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=plot_pt_alpha,
                                markersize=3)

                # For uniform coloring
                # axes[i][j].plot(range(len(all_trials)), all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)

                # For coloring by mean score
                '''perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_mean, perc_cutoff)
                all_trials_below = np.array([x for x in all_trials if np.mean(x)<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials if np.mean(x)>=perc_cutoff_val])
                axes[i][j].plot(range(len(all_trials_below)), all_trials_below, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(range(len(all_trials_mean)-len(all_trials_above), len(all_trials_mean)), all_trials_above, 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)
                '''

                axes[i][j].plot(all_trials_mean, color='black')

                axes[i][j].tick_params(**plot_tick_params)

                '''axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
                axes[i][j].set_ylabel(env_name_title_dict[env_name], **plot_label_params)'''
                if i == 2:
                    # axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                    axes[i][j].set_xlabel('', **plot_label_params)
                if j == 0:
                    axes[i][j].set_ylabel('$S_{a,n,e}$ and $M_{a,n}$\n' + arch_dict_to_label(arch_dict),
                                          **plot_label_params)

        ######################### Plot variance for 2HL4HU
        i = len(arch_dict_list)
        for j, env_name in enumerate(envs_list):
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

            axes[i][j].tick_params(**plot_tick_params)

            axes[i][j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha,
                            markersize=3)

            # axes[i][j].set_xlabel('$M_{a,n}$', **plot_label_params)
            if j == 0:
                axes[i][j].set_ylabel('$V_{a,n}$\n' + arch_dict_to_label(arch_dict), **plot_label_params)

            axes[i][j].set_xlabel('', **plot_label_params)

            # axes[i][j].label_outer()
            '''if i == len(envs_list)-1:
                axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
            '''

        ##################################### Plot histograms for 2HL4HU
        i = len(arch_dict_list) + 1
        for j, env_name in enumerate(envs_list):
            arch_dict = {
                'N_hidden_layers': 2,
                'N_hidden_units': 4
            }
            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]

            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)

            axes[i][j].tick_params(**plot_tick_params)

            axes[i][j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True, bins=20)

            axes[i][j].set_xlabel('$M_{a,n}$\n' + env_name_title_dict[env_name], **plot_label_params)
            if j == 0:
                axes[i][j].set_ylabel('log(Counts)\n' + arch_dict_to_label(arch_dict), **plot_label_params)

            # axes[i][j].label_outer()

        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.97, left=0.09, right=0.99)
        print('Plotting png...')
        plt.savefig(os.path.join(figures_dir, f'combo_plot_v1.png'), dpi=300)

    ###################################### combo plot v2
    if plot_combo_plot_v2:
        print('\nPlotting big combo plot...')

        w_space = 0.3
        fig_width = 8
        adjust_kwargs = {
            'left': 0.08,
            'right': 0.98,
            'top': 0.99
        }
        part_1_h = fig_width / 2
        part_2_h = (2 / 3.0) * part_1_h * 1.2
        plot_tick_params = {
            'axis': 'both',
            'labelsize': 6,
            'which': 'major',
            'pad': 0

        }

        ############################# First part
        plt.close('all')
        N_row = len(arch_dict_list)
        N_col = len(envs_list)
        fig, axes = plt.subplots(N_row, N_col, sharex='col', sharey='col',
                                 gridspec_kw={'hspace': 0, 'wspace': w_space}, figsize=(fig_width, part_1_h))

        ###################### plot main episode/mean plots
        for j, env_name in enumerate(envs_list):
            for i, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting mean and trials of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)

                # For coloring by all episode scores
                all_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_trials)]
                all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
                perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_indexed[:, 1], perc_cutoff)

                all_trials_below = np.array([x for x in all_trials_indexed if x[1] < perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials_indexed if x[1] >= perc_cutoff_val])
                axes[i][j].plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=plot_pt_alpha,
                                markersize=3)
                axes[i][j].plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=plot_pt_alpha,
                                markersize=3)

                # For uniform coloring
                # axes[i][j].plot(range(len(all_trials)), all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)

                # For coloring by mean score
                '''perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_mean, perc_cutoff)
                all_trials_below = np.array([x for x in all_trials if np.mean(x)<perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials if np.mean(x)>=perc_cutoff_val])
                axes[i][j].plot(range(len(all_trials_below)), all_trials_below, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)
                axes[i][j].plot(range(len(all_trials_mean)-len(all_trials_above), len(all_trials_mean)), all_trials_above, 'o', color='mediumseagreen', alpha=plot_pt_alpha, markersize=3)
                '''

                axes[i][j].plot(all_trials_mean, color='black')

                axes[i][j].tick_params(**plot_tick_params)

                '''axes[i][j].set_xlabel(arch_dict_to_label(arch_dict), **plot_label_params)
                axes[i][j].set_ylabel(env_name_title_dict[env_name], **plot_label_params)'''
                if i == 2:
                    # axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                    axes[i][j].set_xlabel('$R_a(n)$', **plot_label_params)
                if j == 0:
                    axes[i][j].set_ylabel('$S_{a,n,e}$ and $M_{a,n}$\n' + arch_dict_to_label(arch_dict),
                                          **plot_label_params)

        plt.subplots_adjust(bottom=0.1, **adjust_kwargs)
        print('Plotting part 1 png...')
        plt.savefig(os.path.join(figures_dir, f'combo_plot_v2_part1.png'), dpi=300)

        ################################ second part
        plt.close('all')
        N_row = 2
        N_col = len(envs_list)
        fig, axes = plt.subplots(N_row, N_col, sharex='col', sharey=False,
                                 gridspec_kw={'hspace': 0, 'wspace': w_space}, figsize=(fig_width, part_2_h))

        ######################### Plot variance for 2HL4HU
        i = 0
        for j, env_name in enumerate(envs_list):
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

            axes[i][j].tick_params(**plot_tick_params)

            axes[i][j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha,
                            markersize=3)

            # axes[i][j].set_xlabel('$M_{a,n}$', **plot_label_params)
            axes[i][j].set_xlabel('', **plot_label_params)
            if j == 0:
                axes[i][j].set_ylabel('$V_{a,n}$\n' + arch_dict_to_label(arch_dict), **plot_label_params)

        ##################################### Plot histograms for 2HL4HU
        i = 1
        for j, env_name in enumerate(envs_list):
            arch_dict = {
                'N_hidden_layers': 2,
                'N_hidden_units': 4
            }
            env_arch_tuple = (env_name, *list(arch_dict.values()))
            print(f'Plotting mean and trials of {env_arch_tuple}...')
            scores = env_arch_score_dict[env_arch_tuple][:10000]

            all_trials = sorted(scores, key=lambda x: np.mean(x))
            all_trials_mean = np.mean(all_trials, axis=1)

            axes[i][j].tick_params(**plot_tick_params)

            axes[i][j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True, bins=20)

            axes[i][j].set_xlabel('$M_{a,n}$\n' + env_name_title_dict[env_name], **plot_label_params)
            if j == 0:
                axes[i][j].set_ylabel('log(Counts)\n' + arch_dict_to_label(arch_dict), **plot_label_params)

        # plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, **adjust_kwargs)
        print('Plotting part 2 png...')
        plt.savefig(os.path.join(figures_dir, f'combo_plot_v2_part2.png'), dpi=300)

    ###################################### combo plot v3
    if plot_combo_plot_v3:
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
            for j, arch_dict in enumerate(arch_dict_list):

                env_arch_tuple = (env_name, *list(arch_dict.values()))
                print(f'Plotting mean and trials of {env_arch_tuple}...')
                scores = env_arch_score_dict[env_arch_tuple][:10000]

                all_trials = sorted(scores, key=lambda x: np.mean(x))
                all_trials_mean = np.mean(all_trials, axis=1)

                # For coloring by all episode scores
                all_trials_indexed = [[[i, y] for y in x] for i, x in enumerate(all_trials)]
                all_trials_indexed = np.array(all_trials_indexed).reshape((-1, 2))
                perc_cutoff = 99.9
                perc_cutoff_val = np.percentile(all_trials_indexed[:, 1], perc_cutoff)

                all_trials_below = np.array([x for x in all_trials_indexed if x[1] < perc_cutoff_val])
                all_trials_above = np.array([x for x in all_trials_indexed if x[1] >= perc_cutoff_val])

                if kwargs.get('mean_lim', None) is None:
                    raise RuntimeError("Did not specify y-limits for mean plot")

                lims = kwargs.get('mean_lim', None)
                axes[i][j].set_ylim(lims[0], lims[1])

                axes[i][j].plot(*all_trials_below.transpose(), 'o', color='tomato', alpha=plot_pt_alpha,
                                markersize=3)
                axes[i][j].plot(*all_trials_above.transpose(), 'o', color='mediumseagreen', alpha=plot_pt_alpha,
                                markersize=3)

                # For uniform coloring
                # axes[i][j].plot(range(len(all_trials)), all_trials, 'o', color='tomato', alpha=plot_pt_alpha, markersize=3)

                axes[i][j].plot(all_trials_mean, color='black')

                axes[i][j].tick_params(**plot_tick_params)

                # arch_dict_to_label(arch_dict)
                if i == len(envs_list) - 1:
                    # axes[i][j].set_xlabel(env_name_title_dict[env_name], **plot_label_params)
                    axes[i][j].set_xlabel('$R_a(n)$,\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)
                if j == 0:
                    axes[i][j].set_ylabel(env_name_title_dict[env_name] + '\n\n$S_{a,n,e}$ and $M_{a,n}$',
                                          **plot_label_params)

        plt.subplots_adjust(left=0.2, **adjust_kwargs)
        print('Plotting part 1 png...')
        part_1_path = os.path.join(figures_dir, f'combo_part1.png')
        plt.savefig(part_1_path, dpi=300)

        ################################ second part
        plt.close('all')
        N_row = len(envs_list)
        N_col = 2
        fig, axes = plt.subplots(N_row, N_col, sharex=False, sharey=False,
                                 gridspec_kw={'hspace': 0, 'wspace': 0.23}, figsize=(part_2_w, fig_height))

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
            axes[i][j].set_ylim(lims[0], lims[1])

            if kwargs.get('mean_lim', None) is None:
                raise RuntimeError("Did not specify x-limits for variance plot")

            lims = kwargs.get('mean_lim', None)
            axes[i][j].set_xlim(lims[0], lims[1])

            axes[i][j].tick_params(**plot_tick_params)
            axes[i][j].tick_params(axis='x', labelsize=0)

            axes[i][j].plot(all_trials_mean, all_trials_std, 'o', color='mediumorchid', alpha=plot_pt_alpha,
                            markersize=3)

            # axes[i][j].set_xlabel('$M_{a,n}$', **plot_label_params)
            axes[i][j].set_ylabel('$V_{a,n}$', **plot_label_params)
            if i == len(envs_list) - 1:
                axes[i][j].set_xlabel('$M_{a,n}$\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)

            # axes[i][j].label_outer()

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
            axes[i][j].set_ylim(lims[0], lims[1])

            if kwargs.get('N_bins', None) is None:
                raise RuntimeError("Did not specify bins for histogram")

            axes[i][j].hist(all_trials_mean, color='dodgerblue', edgecolor='gray', log=True, bins=kwargs.get('N_bins', None))
            axes[i][j].tick_params(**plot_tick_params)
            axes[i][j].tick_params(axis='x', labelsize=0)

            axes[i][j].set_ylabel('', **plot_label_params)
            if i == len(envs_list) - 1:
                axes[i][j].set_xlabel('$M_{a,n}$\n\n' + arch_dict_to_label(arch_dict), **plot_label_params)

            # axes[i][j].label_outer()

        # plt.tight_layout()
        plt.subplots_adjust(left=0.2, **adjust_kwargs)
        print('Plotting part 2 png...')
        part_2_path = os.path.join(figures_dir, f'combo_part2.png')
        plt.savefig(part_2_path, dpi=300)

        # Save combined plots
        im1 = cv2.imread(part_1_path)
        im2 = cv2.imread(part_2_path)
        combined_path = os.path.join(figures_dir, f'combo_full.png')
        combined_plot = cv2.hconcat([im1, im2])
        cv2.imwrite(combined_path, combined_plot)


