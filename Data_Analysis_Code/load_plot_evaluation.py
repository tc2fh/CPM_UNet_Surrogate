'''
Use utility functions from evaluate_stats_functions.py to load and plot evaluation metrics from saved data.

data generated from run_evaluation.py
'''
#%% imports
import sys
import evaluate_stats_functions as esf
import os
import time

#%% functions

saved_stats_parent_directory = r"" #directory where run_evaluation data was saved to

def plot_weighted_metrics(metric_str, ylabel_str=None, xlabel_str=None, model_legend_str='Surrogate Prediction', h0_label_str='Reference Configuration', short_timescale_range=None, semilogy=False, saved_stats_parent_directory=saved_stats_parent_directory, title_str = False, title_size = 20, label_size = 15, tick_size = 15, legend_size = 10, print_data=False, prediction_color='blue', h0_color='orange'):
    ''' plots just the not split model, it looked the best at longer timescales in the video '''
    saved_stats_parent_directory = saved_stats_parent_directory
    pdf_options = ['PDF', 'true_hist']
    bins_options = ['fd', 'sturges']
    model_types = ['periodic_split_weighted', 'periodic_weighted']
    model_names = ['bce_2loss_1_10', 'dice_2loss_1_10']

    # plot periodic_weighted bce_2loss_1_10 PDF sturges
    stats_dir = os.path.join(saved_stats_parent_directory, pdf_options[0], bins_options[1], model_types[1], model_names[0])
    setup_dir = esf.setup_directories(stats_dir, make_directories=False)
    metrics = esf.load_all_metrics(setup_dir, TryDomainHist=True)
    if ylabel_str:
        ylabel_string = ylabel_str
    else:
        ylabel_string = False
    prediction, h0, model_prediction_std, h0_std = esf.plot_single_metric(metric=metrics[metric_str],model_legend_str=model_legend_str, h0_label_str=h0_label_str, metric_name=metric_str.upper(), ylabel=ylabel_string, xlabel=xlabel_str, short_timescale=short_timescale_range, semilogy=semilogy, title_str=title_str, title_size=title_size, label_size=label_size, tick_size=tick_size, legend_size=legend_size, prediction_color_str=prediction_color, h0_color_str=h0_color)
    if print_data:
        print('prediction:', prediction)
        print('h0:', h0)
        print('model_prediction_std:', model_prediction_std)
        print('h0_std:', h0_std)


'''
data series:
dice
mse
median_se
percentile_se
sum_chem_se
sum_mask_se
chem_cell_compare_se
frac_cells_se
sum_vegf_field
sum_cell_seg
difference_sum_vegf
difference_sum_cell_seg
area_emd
area_mean_se
area_median_se
width_emd
width_mean_se
width_median_se
'''

# %%
short_timsecale_range = (0,9)
metric_str = 'dice'

surrogate_color = 'lightseagreen'
h0_color = 'black'
plot_weighted_metrics(metric_str, ylabel_str='Dice Score', short_timescale_range=short_timsecale_range, semilogy=False, title_str=False, label_size=25, tick_size=20, legend_size=15, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
plot_weighted_metrics(metric_str, ylabel_str='Dice Score', xlabel_str='Timestep (MCS)',short_timescale_range=None, semilogy=False, title_str=None, label_size=25, tick_size=20, legend_size=None, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
print('picture in picture')
plot_weighted_metrics(metric_str, short_timescale_range=short_timsecale_range, semilogy=False, title_str=None, label_size=25, tick_size=30, legend_size=None, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
plot_weighted_metrics(metric_str, ylabel_str='Dice', xlabel_str='MCS', short_timescale_range=short_timsecale_range, semilogy=False, title_str=None, label_size=30, tick_size=30, legend_size=None, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
metric_str = 'mse'
plot_weighted_metrics(metric_str, ylabel_str='Diffusive Field MSE', short_timescale_range=short_timsecale_range, semilogy=False, title_str=False, label_size=25, tick_size=20, legend_size=15, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
plot_weighted_metrics(metric_str, ylabel_str='Diffusive Field MSE', xlabel_str='Timestep (MCS)', short_timescale_range=None, semilogy=False, title_str=None, label_size=25, tick_size=20, legend_size=None, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
print('picture in picture')
plot_weighted_metrics(metric_str, short_timescale_range=short_timsecale_range, semilogy=False, title_str=False, label_size=25, tick_size=30, legend_size=None, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
plot_weighted_metrics(metric_str, ylabel_str='Diffusive Field MSE', xlabel_str='Timestep (MCS)', short_timescale_range=short_timsecale_range, semilogy=False, title_str=False, label_size=30, tick_size=30, legend_size=23, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)

print('print initial data points and stdv')
plot_weighted_metrics('dice', ylabel_str='Dice', xlabel_str='MCS', short_timescale_range=short_timsecale_range, semilogy=False, title_str=None, label_size=30, tick_size=30, legend_size=None, print_data=True, prediction_color=surrogate_color, h0_color=h0_color)
# plot_weighted_metrics(metric_str, ylabel_str='Diffusive Field MSE', xlabel_str='Timestep (MCS)', short_timescale_range=short_timsecale_range, semilogy=False, title_str=False, label_size=30, tick_size=30, legend_size=23, print_data=True, prediction_color=surrogate_color, h0_color=h0_color)


#%%
h0_color_cellseg = 'black'
surrogate_color_cellseg = 'tab:purple'
metric_str = 'sum_cell_seg'
model_legend_str = 'Surrogate Prediction'
h0_label_str = 'Ground Truth' # for this data series and sum_vegf_field, the second series saved is the ground truth at the MCS equivalent to the surrogate prediction
plot_weighted_metrics(metric_str, ylabel_str='Total Vessel Area', xlabel_str='Timestep (MCS)',short_timescale_range=None, semilogy=False, title_str=None, label_size=25, tick_size=20, legend_size=None, print_data=False, prediction_color=surrogate_color_cellseg, h0_color=h0_color_cellseg)

metric_str = 'sum_vegf_field'
plot_weighted_metrics(metric_str, ylabel_str='Diffusive Field Sum', xlabel_str='Timestep (MCS)',model_legend_str=model_legend_str, h0_label_str=h0_label_str, short_timescale_range=None, semilogy=False, title_str=None, label_size=25, tick_size=20, legend_size=15, print_data=False, prediction_color=surrogate_color_cellseg, h0_color=h0_color_cellseg)

# %%
metric_str = 'area_emd'
surrogate_color = 'lightseagreen'
h0_color = 'black'
plot_weighted_metrics(metric_str, ylabel_str='Lacunae Area EMD', xlabel_str='Timestep (MCS)',short_timescale_range=short_timsecale_range, semilogy=False, title_str=None, label_size=30, tick_size=30, legend_size=None, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
plot_weighted_metrics(metric_str, ylabel_str='Lacunae Area EMD', xlabel_str='Timestep (MCS)',short_timescale_range=None, semilogy=False, title_str=None, label_size=25, tick_size=20, legend_size=None, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)

plot_weighted_metrics(metric_str, ylabel_str='Lacunae Area EMD', xlabel_str='Timestep (MCS)',short_timescale_range=None, semilogy=False, title_str=None, label_size=25, tick_size=20, legend_size=20, print_data=False, prediction_color=surrogate_color, h0_color=h0_color)
tiny_timescale_range = (0,3)
plot_weighted_metrics(metric_str, ylabel_str='Lacunae Area EMD', xlabel_str='Timestep (MCS)',short_timescale_range=tiny_timescale_range, semilogy=False, title_str=None, label_size=25, tick_size=20, legend_size=20, print_data=True, prediction_color=surrogate_color, h0_color=h0_color)

# %%
