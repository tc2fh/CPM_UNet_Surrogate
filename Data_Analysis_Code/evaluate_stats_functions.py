'''
utility functions for loading a model, calculate statistics after running it on a large dataset

use with run_evaluation.py 
'''

#%% imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from Periodic_Unet_2_loss import LitModelPeriodic as periodic_unet_2_loss
from Periodic_Unet_3_loss import LitModelPeriodic as periodic_unet_3_loss
from Periodic_Unet_4_loss import LitModelPeriodic as periodic_unet_4_loss
from Split_Periodic_2loss import LitModelPeriodic as split_periodic_2_loss
from recursive_split_Periodic_2loss import LitModelPeriodic as recursive_split_periodic_2_loss
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import zarr
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from scipy.stats import wasserstein_distance
from skimage.morphology import medial_axis, skeletonize
from concurrent.futures import ProcessPoolExecutor
import glob

#%% functions for loading data, passing through model, evaluating model, calculating stats
def load_data(zipstore_path, timestep_index, timestep_ahead=10):
    '''Load the data from the zipstore at the given timestep index and at index + 10 for the 10 timesteps ahead'''
    with zarr.open(store=zarr.storage.ZipStore(zipstore_path, mode="r")) as root:
        fgbg = np.array(root["fgbg"][timestep_index])  
        vegf = np.array(root["vegf"][timestep_index])  
        fgbg_next = np.array(root["fgbg"][timestep_index + timestep_ahead])  
        vegf_next = np.array(root["vegf"][timestep_index + timestep_ahead])  
    return fgbg, vegf, fgbg_next, vegf_next

def pass_through_model(model, input_stack,threshold=0.5,probabilities=False):
    '''Pass the input stack through the model and return the output, sigmoided and thresholded'''
    input_tensor = torch.from_numpy(input_stack).float().unsqueeze(0).permute(0, 3, 1, 2) #convert to tensor with shape (1, 2, 256, 256), (batch, channels, height, width)

    # Move the input tensor to the same device as the model
    input_tensor = input_tensor.to(model.device)

    # Ensure the model is in evaluation mode
    model.eval()

    # Disable gradient calculations
    with torch.no_grad():
        # Pass the input tensor through the model
        output = model(input_tensor) # output is [batch_size, 2, 256, 256]

    #sigmoid the cell (segmentation) channel
    if probabilities:
        output[:,0,:,:] = torch.sigmoid(output[:,0,:,:]) #sigmoided for probabilities
    else:
        output[:,0,:,:] = torch.sigmoid(output[:,0,:,:])
        output[:,0,:,:] = (output[:,0,:,:]>threshold).int() #thresholded probabilities to binary mask

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy() #convert back to numpy with shape (256, 256, 2)
    return output

def evaluate_model_with_hist(timestep, evaluations, model, zipstore_path, bins_option='auto', hist_PDF=True, timestep_ahead=100, stats_percentile=75,plot=False, print_iter=False):
    '''
    in this function vs evaluate_model: histogram metrics included in the evaluation metrics so that model inference only needs to happen once
    Evaluate the model at a given timestep, append evaluation metrics to arrays
    as it propagates for a number of iterations, plot the dice and mse

    evaluation metrics:
    dice coefficient
    avg mean squared error
    median mean squared error
    75th percentile of the squared error (editable in the function call)

    sum chem field squared error
    sum cell mask squarred error
    sum chem field/sum cell mask squarred error (chem_cell_compare)
    fraction occupied by cells (sum cell mask/total pixels-256*256) squared error (cell_fraction)

    inputs:
    timestep: int, the timestep to start the evaluation
    evaluations: int, the number of iterations to evaluate the model
    model: torch model, the model to evaluate
    zipstore_path: str, the path to the zipstore containing the data
    bins_option: str, the option for the number of bins in the histogram, 'auto' for automatic binning with freedman-diaconis numpy default method
    hist_PDF: bool, if True, plot the histogram as a probability density function
    timestep_ahead: int, the number of timesteps ahead that the model was trained to predict
    stats_percentile: int, the percentile to use for the mse statistics
    plot: bool, if True, plot the dice and error statistics
    print_iter: bool, if True, print the evaluation iteration number
    '''
    font_size = 20
    input_stack = None
    initial_fgbg, initial_vegf, initial_fgbg_next, initial_vegf_next = load_data(zipstore_path, timestep, timestep_ahead=timestep_ahead)
    
    #store arrays for plotting vs null hypothesis (no change from input to output)
    dice_coeff_array = []
    dice_coeff_original_array = []

    avg_mse_prediction_array = []
    avg_mse_input_vs_gt_array = []

    median_mse_prediction_array = []
    median_mse_input_vs_gt_array = []

    percentile_mse_prediction_array = []
    percentile_mse_input_vs_gt_array = []

    sum_chem_field_prediction_array = [] # squared error
    sum_chem_field_input_vs_gt_array = []

    sum_cell_mask_prediction_array = [] # squared error
    sum_cell_mask_input_vs_gt_array = []

    chem_cell_compare_prediction_array = [] # squared error
    chem_cell_compare_input_vs_gt_array = []

    cell_fraction_prediction_array = [] # squared error
    cell_fraction_input_vs_gt_array = []

    sum_vegf_field_prediction_array = [] #not squared error
    sum_vegf_field_input_vs_gt_array = []

    sum_cell_seg_prediction_array = [] #not squared error
    sum_cell_seg_input_vs_gt_array = []

    difference_sum_vegf_field_prediction = [] #not squared error
    difference_sum_vegf_field_input_vs_gt = []

    difference_sum_cell_seg_prediction = [] #not squared error
    difference_sum_cell_seg_input_vs_gt = []

    area_emd_pred_gt_array = []
    area_emd_nh_gt_array = []

    mean_se_areas_pred_gt_array = []
    mean_se_areas_nh_gt_array = []

    median_se_areas_pred_gt_array = []
    median_se_areas_nh_gt_array = []

    width_emd_pred_gt_array = []
    width_emd_nh_gt_array = []

    mean_se_width_pred_gt_array = []
    mean_se_width_nh_gt_array = []

    median_se_width_pred_gt_array = []
    median_se_width_nh_gt_array = []

    timestep_array = []

    for i in range(evaluations):
        if print_iter:
            print("iteration ", i+1, "of ", evaluations)
        # Load data
        fgbg, vegf, fgbg_next, vegf_next = load_data(zipstore_path, timestep + i*timestep_ahead, timestep_ahead=timestep_ahead)
        if i == 0:
            input_stack = np.stack([fgbg, vegf], axis=-1)
            timestep_array.append(timestep + timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        else:
            input_stack = np.stack([output[:,:,0], output[:,:,1]], axis=-1)  # Use the previous output as input
            timestep_array.append(timestep + i*timestep_ahead + timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        if input_stack is None:
            raise ValueError("input_stack is None")

        # Pass through model
        output = pass_through_model(model, input_stack)
        
        dice_coeff = (2.0 * np.sum(fgbg_next * output[:,:,0])) / (np.sum(fgbg_next) + np.sum(output[:,:,0]))
        dice_coeff_array.append(dice_coeff)

        dice_original = (2.0 * np.sum(initial_fgbg * fgbg_next)) / (np.sum(initial_fgbg) + np.sum(fgbg_next))
        dice_coeff_original_array.append(dice_original)

        avg_mse_pred = np.mean((vegf_next - output[:,:,1])**2)
        avg_mse_prediction_array.append(avg_mse_pred)

        avg_mse_input = np.mean((vegf_next - initial_vegf)**2)
        avg_mse_input_vs_gt_array.append(avg_mse_input)

        median_mse_pred = np.median((vegf_next - output[:,:,1])**2)
        median_mse_prediction_array.append(median_mse_pred)

        median_mse_input = np.median((vegf_next - initial_vegf)**2)
        median_mse_input_vs_gt_array.append(median_mse_input)

        percentile_mse_pred = np.percentile((vegf_next - output[:,:,1])**2, stats_percentile)  # top 25th percentile if set to 75
        percentile_mse_prediction_array.append(percentile_mse_pred)

        percentile_mse_input = np.percentile((vegf_next - initial_vegf)**2, stats_percentile)  # top 25th percentile if set to 75
        percentile_mse_input_vs_gt_array.append(percentile_mse_input)

        sum_chem_field_prediction = (np.sum(output[:,:,1])-np.sum(vegf_next))**2
        sum_chem_field_prediction_array.append(sum_chem_field_prediction)

        sum_chem_field_input_vs_gt = (np.sum(initial_vegf)-np.sum(vegf_next))**2
        sum_chem_field_input_vs_gt_array.append(sum_chem_field_input_vs_gt)

        sum_cell_mask_prediction = (np.sum(output[:,:,0])-np.sum(fgbg_next))**2
        sum_cell_mask_prediction_array.append(sum_cell_mask_prediction)

        sum_cell_mask_input_vs_gt = (np.sum(initial_fgbg)-np.sum(fgbg_next))**2
        sum_cell_mask_input_vs_gt_array.append(sum_cell_mask_input_vs_gt)
        
        chem_cell_compare_prediction = (np.sum(output[:,:,1])/np.sum(output[:,:,0]) - np.sum(vegf_next)/np.sum(fgbg_next))**2
        chem_cell_compare_prediction_array.append(chem_cell_compare_prediction)

        chem_cell_compare_input_vs_gt = (np.sum(initial_vegf)/np.sum(initial_fgbg) - np.sum(vegf_next)/np.sum(fgbg_next))**2
        chem_cell_compare_input_vs_gt_array.append(chem_cell_compare_input_vs_gt)

        cell_fraction_prediction = (np.sum(output[:,:,0])/(256*256) - np.sum(fgbg_next)/(256*256))**2
        cell_fraction_prediction_array.append(cell_fraction_prediction)

        cell_fraction_input_vs_gt = (np.sum(initial_fgbg)/(256*256) - np.sum(fgbg_next)/(256*256))**2
        cell_fraction_input_vs_gt_array.append(cell_fraction_input_vs_gt)

        sum_vegf_field_prediction = np.sum(output[:,:,1])
        sum_vegf_field_prediction_array.append(sum_vegf_field_prediction)

        sum_vegf_field_input_vs_gt = np.sum(vegf_next)
        sum_vegf_field_input_vs_gt_array.append(sum_vegf_field_input_vs_gt)

        sum_cell_seg_prediction = np.sum(output[:,:,0])
        sum_cell_seg_prediction_array.append(sum_cell_seg_prediction)

        sum_cell_seg_input_vs_gt = np.sum(fgbg_next)
        sum_cell_seg_input_vs_gt_array.append(sum_cell_seg_input_vs_gt)

        difference_sum_vegf_field_pred = np.sum(output[:,:,1]) - np.sum(vegf_next)
        difference_sum_vegf_field_prediction.append(difference_sum_vegf_field_pred)

        difference_sum_vegf_field_val = np.sum(initial_vegf) - np.sum(vegf_next)
        difference_sum_vegf_field_input_vs_gt.append(difference_sum_vegf_field_val)

        difference_sum_cell_seg_pred = np.sum(output[:,:,0]) - np.sum(fgbg_next)
        difference_sum_cell_seg_prediction.append(difference_sum_cell_seg_pred)

        difference_sum_cell_seg__val = np.sum(initial_fgbg) - np.sum(fgbg_next)
        difference_sum_cell_seg_input_vs_gt.append(difference_sum_cell_seg__val)

        # Calculate the histogram statistics for area
        pred_areas, gt_areas, ho_areas, bins, emd_pred_gt, emd_nh_gt = domain_area_periodic_histogram(output[:,:,0], fgbg_next, initial_fgbg, bins_option=bins_option, plot=False, distribution_distance=True, PDF=hist_PDF)
        area_emd_pred_gt_array.append(emd_pred_gt)
        area_emd_nh_gt_array.append(emd_nh_gt)

        mean_pred_areas = np.mean(pred_areas)
        mean_gt_areas = np.mean(gt_areas)
        mean_ho_areas = np.mean(ho_areas)

        median_pred_areas = np.median(pred_areas)
        median_gt_areas = np.median(gt_areas)
        median_ho_areas = np.median(ho_areas)

        mean_se_areas_pred_gt_array.append((mean_pred_areas-mean_gt_areas)**2)
        mean_se_areas_nh_gt_array.append((mean_ho_areas-mean_gt_areas)**2)

        median_se_areas_pred_gt_array.append((median_pred_areas-median_gt_areas)**2)
        median_se_areas_nh_gt_array.append((median_ho_areas-median_gt_areas)**2)

        # Calculate the histogram statistics
        pred_widths, gt_widths, ho_widths, bins, emd_pred_gt, emd_nh_gt = width_distribution_histograms(output[:,:,0], fgbg_next, initial_fgbg, bins_option=bins_option, plot=False, distribution_distance=True, PDF=hist_PDF)
        width_emd_pred_gt_array.append(emd_pred_gt)
        width_emd_nh_gt_array.append(emd_nh_gt)

        mean_pred_widths = np.mean(pred_widths)
        mean_gt_widths = np.mean(gt_widths)
        mean_ho_widths = np.mean(ho_widths)

        median_pred_widths = np.median(pred_widths)
        median_gt_widths = np.median(gt_widths)
        median_ho_widths = np.median(ho_widths)

        mean_se_width_pred_gt_array.append((mean_pred_widths-mean_gt_widths)**2)
        mean_se_width_nh_gt_array.append((mean_ho_widths-mean_gt_widths)**2)

        median_se_width_pred_gt_array.append((median_pred_widths-median_gt_widths)**2)
        median_se_width_nh_gt_array.append((median_ho_widths-median_gt_widths)**2)

    return (
        dice_coeff_array, 
        dice_coeff_original_array,
        avg_mse_prediction_array, 
        avg_mse_input_vs_gt_array,
        median_mse_prediction_array, 
        median_mse_input_vs_gt_array, 
        percentile_mse_prediction_array, 
        percentile_mse_input_vs_gt_array,
        sum_chem_field_prediction_array,
        sum_chem_field_input_vs_gt_array,
        sum_cell_mask_prediction_array,
        sum_cell_mask_input_vs_gt_array,
        chem_cell_compare_prediction_array,
        chem_cell_compare_input_vs_gt_array,
        cell_fraction_prediction_array,
        cell_fraction_input_vs_gt_array,
        sum_vegf_field_prediction_array,
        sum_vegf_field_input_vs_gt_array,
        sum_cell_seg_prediction_array,
        sum_cell_seg_input_vs_gt_array,
        difference_sum_vegf_field_prediction,
        difference_sum_vegf_field_input_vs_gt,
        difference_sum_cell_seg_prediction,
        difference_sum_cell_seg_input_vs_gt,
        timestep_array,
        area_emd_pred_gt_array,
        area_emd_nh_gt_array,
        mean_se_areas_pred_gt_array,
        mean_se_areas_nh_gt_array,
        median_se_areas_pred_gt_array,
        median_se_areas_nh_gt_array,
        width_emd_pred_gt_array,
        width_emd_nh_gt_array,
        mean_se_width_pred_gt_array,
        mean_se_width_nh_gt_array,
        median_se_width_pred_gt_array,
        median_se_width_nh_gt_array
    )

def evaluate_model(timestep, evaluations, model, zipstore_path,timestep_ahead=100, stats_percentile=75,plot=False, print_iter=False):
    '''
    Evaluate the model at a given timestep, append evaluation metrics to arrays
    as it propagates for a number of iterations, plot the dice and mse

    evaluation metrics:
    dice coefficient
    avg mean squared error
    median mean squared error
    75th percentile of the squared error (editable in the function call)

    sum chem field squared error
    sum cell mask squarred error
    sum chem field/sum cell mask squarred error (chem_cell_compare)
    fraction occupied by cells (sum cell mask/total pixels-256*256) squared error (cell_fraction)


    inputs:
    timestep: int, the timestep to start the evaluation
    evaluations: int, the number of iterations to evaluate the model
    model: torch model, the model to evaluate
    zipstore_path: str, the path to the zipstore containing the data
    timestep_ahead: int, the number of timesteps ahead that the model was trained to predict
    stats_percentile: int, the percentile to use for the mse statistics
    plot: bool, if True, plot the dice and error statistics
    print_iter: bool, if True, print the evaluation iteration number
    '''
    font_size = 20
    input_stack = None
    initial_fgbg, initial_vegf, initial_fgbg_next, initial_vegf_next = load_data(zipstore_path, timestep, timestep_ahead=timestep_ahead)
    
    #store arrays for plotting vs null hypothesis (no change from input to output)
    dice_coeff_array = []
    dice_coeff_original_array = []

    avg_mse_prediction_array = []
    avg_mse_input_vs_gt_array = []

    median_mse_prediction_array = []
    median_mse_input_vs_gt_array = []

    percentile_mse_prediction_array = []
    percentile_mse_input_vs_gt_array = []

    sum_chem_field_prediction_array = [] # squared error
    sum_chem_field_input_vs_gt_array = []

    sum_cell_mask_prediction_array = [] # squared error
    sum_cell_mask_input_vs_gt_array = []

    chem_cell_compare_prediction_array = [] # squared error
    chem_cell_compare_input_vs_gt_array = []

    cell_fraction_prediction_array = [] # squared error
    cell_fraction_input_vs_gt_array = []

    sum_vegf_field_prediction_array = [] #not squared error
    sum_vegf_field_input_vs_gt_array = []

    sum_cell_seg_prediction_array = [] #not squared error
    sum_cell_seg_input_vs_gt_array = []

    difference_sum_vegf_field_prediction = [] #not squared error
    difference_sum_vegf_field_input_vs_gt = []

    difference_sum_cell_seg_prediction = [] #not squared error
    difference_sum_cell_seg_input_vs_gt = []

    timestep_array = []

    for i in range(evaluations):
        if print_iter:
            print("iteration ", i+1, "of ", evaluations)
        # Load data
        fgbg, vegf, fgbg_next, vegf_next = load_data(zipstore_path, timestep + i*timestep_ahead, timestep_ahead=timestep_ahead)
        if i == 0:
            input_stack = np.stack([fgbg, vegf], axis=-1)
            timestep_array.append(timestep + timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        else:
            input_stack = np.stack([output[:,:,0], output[:,:,1]], axis=-1)  # Use the previous output as input
            timestep_array.append(timestep + i*timestep_ahead + timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        if input_stack is None:
            raise ValueError("input_stack is None")

        # Pass through model
        output = pass_through_model(model, input_stack)
        
        dice_coeff = (2.0 * np.sum(fgbg_next * output[:,:,0])) / (np.sum(fgbg_next) + np.sum(output[:,:,0]))
        dice_coeff_array.append(dice_coeff)

        dice_original = (2.0 * np.sum(initial_fgbg * fgbg_next)) / (np.sum(initial_fgbg) + np.sum(fgbg_next))
        dice_coeff_original_array.append(dice_original)

        avg_mse_pred = np.mean((vegf_next - output[:,:,1])**2)
        avg_mse_prediction_array.append(avg_mse_pred)

        avg_mse_input = np.mean((vegf_next - initial_vegf)**2)
        avg_mse_input_vs_gt_array.append(avg_mse_input)

        median_mse_pred = np.median((vegf_next - output[:,:,1])**2)
        median_mse_prediction_array.append(median_mse_pred)

        median_mse_input = np.median((vegf_next - initial_vegf)**2)
        median_mse_input_vs_gt_array.append(median_mse_input)

        percentile_mse_pred = np.percentile((vegf_next - output[:,:,1])**2, stats_percentile)  # top 25th percentile if set to 75
        percentile_mse_prediction_array.append(percentile_mse_pred)

        percentile_mse_input = np.percentile((vegf_next - initial_vegf)**2, stats_percentile)  # top 25th percentile if set to 75
        percentile_mse_input_vs_gt_array.append(percentile_mse_input)

        sum_chem_field_prediction = (np.sum(output[:,:,1])-np.sum(vegf_next))**2
        sum_chem_field_prediction_array.append(sum_chem_field_prediction)

        sum_chem_field_input_vs_gt = (np.sum(initial_vegf)-np.sum(vegf_next))**2
        sum_chem_field_input_vs_gt_array.append(sum_chem_field_input_vs_gt)

        sum_cell_mask_prediction = (np.sum(output[:,:,0])-np.sum(fgbg_next))**2
        sum_cell_mask_prediction_array.append(sum_cell_mask_prediction)

        sum_cell_mask_input_vs_gt = (np.sum(initial_fgbg)-np.sum(fgbg_next))**2
        sum_cell_mask_input_vs_gt_array.append(sum_cell_mask_input_vs_gt)
        
        chem_cell_compare_prediction = (np.sum(output[:,:,1])/np.sum(output[:,:,0]) - np.sum(vegf_next)/np.sum(fgbg_next))**2
        chem_cell_compare_prediction_array.append(chem_cell_compare_prediction)

        chem_cell_compare_input_vs_gt = (np.sum(initial_vegf)/np.sum(initial_fgbg) - np.sum(vegf_next)/np.sum(fgbg_next))**2
        chem_cell_compare_input_vs_gt_array.append(chem_cell_compare_input_vs_gt)

        cell_fraction_prediction = (np.sum(output[:,:,0])/(256*256) - np.sum(fgbg_next)/(256*256))**2
        cell_fraction_prediction_array.append(cell_fraction_prediction)

        cell_fraction_input_vs_gt = (np.sum(initial_fgbg)/(256*256) - np.sum(fgbg_next)/(256*256))**2
        cell_fraction_input_vs_gt_array.append(cell_fraction_input_vs_gt)

        sum_vegf_field_prediction = np.sum(output[:,:,1])
        sum_vegf_field_prediction_array.append(sum_vegf_field_prediction)

        sum_vegf_field_input_vs_gt = np.sum(vegf_next)
        sum_vegf_field_input_vs_gt_array.append(sum_vegf_field_input_vs_gt)

        sum_cell_seg_prediction = np.sum(output[:,:,0])
        sum_cell_seg_prediction_array.append(sum_cell_seg_prediction)

        sum_cell_seg_input_vs_gt = np.sum(fgbg_next)
        sum_cell_seg_input_vs_gt_array.append(sum_cell_seg_input_vs_gt)

        difference_sum_vegf_field_pred = np.sum(output[:,:,1]) - np.sum(vegf_next)
        difference_sum_vegf_field_prediction.append(difference_sum_vegf_field_pred)

        difference_sum_vegf_field_val = np.sum(initial_vegf) - np.sum(vegf_next)
        difference_sum_vegf_field_input_vs_gt.append(difference_sum_vegf_field_val)

        difference_sum_cell_seg_pred = np.sum(output[:,:,0]) - np.sum(fgbg_next)
        difference_sum_cell_seg_prediction.append(difference_sum_cell_seg_pred)

        difference_sum_cell_seg__val = np.sum(initial_fgbg) - np.sum(fgbg_next)
        difference_sum_cell_seg_input_vs_gt.append(difference_sum_cell_seg__val)


    if plot == True:
        # subplots with dice coeff, mse, median mse, top 25th percentile mse
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2 rows, 2 columns

        # Dice Coefficient
        axs[0, 0].plot(timestep_array, dice_coeff_array, 'o--', label='model prediction')
        axs[0, 0].plot(timestep_array, dice_coeff_original_array, 'o--', label='null hypothesis')
        axs[0, 0].set_title('Dice Coefficient (cell mask)')
        axs[0, 0].set_xlabel('timestep')
        axs[0, 0].set_ylabel('Dice')
        axs[0, 0].legend()

        # Avg Mean Square Error
        axs[0, 1].plot(timestep_array, avg_mse_prediction_array, 'o--', label='model prediction')
        axs[0, 1].plot(timestep_array, avg_mse_input_vs_gt_array, 'o--', label='null hypothesis')
        axs[0, 1].set_title('MSE (vegf)')
        axs[0, 1].set_xlabel('timestep')
        axs[0, 1].set_ylabel('Error')
        axs[0, 1].legend()

        # Median Mean Square Error
        axs[1, 0].plot(timestep_array, median_mse_prediction_array, 'o--', label='model prediction')
        axs[1, 0].plot(timestep_array, median_mse_input_vs_gt_array, 'o--', label='null hypothesis')
        axs[1, 0].set_title('Median SE (vegf)')
        axs[1, 0].set_xlabel('timestep')
        axs[1, 0].set_ylabel('Error')
        axs[1, 0].legend()

        # Top 25 Mean Square Error
        axs[1, 1].plot(timestep_array, percentile_mse_prediction_array, 'o--', label='model prediction')
        axs[1, 1].plot(timestep_array, percentile_mse_input_vs_gt_array, 'o--', label='null hypothesis')
        axs[1, 1].set_title(f'{stats_percentile} Percentile SE (vegf)')
        axs[1, 1].set_xlabel('timestep')
        axs[1, 1].set_ylabel('Error')
        axs[1, 1].legend()

        plt.suptitle(f'Original Metrics for timestep {timestep}, {evaluations} evaluations', fontsize=20)

        plt.tight_layout()
        plt.show()

        # subplots with sum chem field squared error, sum cell mask squared error, chem_cell_compare, cell_fraction
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2 rows, 2 columns

        # Sum cell mask squared error
        axs[0, 0].plot(timestep_array, sum_cell_mask_prediction_array, 'o--', label='model prediction')
        axs[0, 0].plot(timestep_array, sum_cell_mask_input_vs_gt_array, 'o--', label='null hypothesis')
        axs[0, 0].set_title(r'$\Sigma$Cell Mask SE')
        axs[0, 0].set_xlabel('timestep')
        axs[0, 0].set_ylabel('Error')
        axs[0, 0].legend()


        # Sum Chem Field Squared Error
        axs[0, 1].plot(timestep_array, sum_chem_field_prediction_array, 'o--', label='model prediction')
        axs[0, 1].plot(timestep_array, sum_chem_field_input_vs_gt_array, 'o--', label='null hypothesis')
        axs[0, 1].set_title(r'$\Sigma$VEGF Field SE')
        axs[0, 1].set_xlabel('timestep')
        axs[0, 1].set_ylabel('Error')
        axs[0, 1].legend()
        
        # chem cell compare
        axs[1, 0].plot(timestep_array, chem_cell_compare_prediction_array, 'o--', label='model prediction')
        axs[1, 0].plot(timestep_array, chem_cell_compare_input_vs_gt_array, 'o--', label='null hypothesis')
        axs[1, 0].set_title(r'$\Sigma$VEGF field/$\Sigma$Cell mask SE')
        axs[1, 0].set_xlabel('timestep')
        axs[1, 0].set_ylabel('Error')
        axs[1, 0].legend()
        
        # cell fraction
        axs[1, 1].plot(timestep_array, cell_fraction_prediction_array, 'o--', label='model prediction')
        axs[1, 1].plot(timestep_array, cell_fraction_input_vs_gt_array, 'o--', label='null hypothesis')
        axs[1, 1].set_title(r'Cell Fraction ($\Sigma$mask/total pixels) SE')
        axs[1, 1].set_xlabel('timestep')
        axs[1, 1].set_ylabel('Error')
        axs[1, 1].legend()

        plt.suptitle(f'Additional Metrics for timestep {timestep}, {evaluations} evaluations', fontsize=20)

        plt.tight_layout()
        plt.show()

        return 
    else:
        return (
            dice_coeff_array, 
            dice_coeff_original_array,
            avg_mse_prediction_array, 
            avg_mse_input_vs_gt_array,
            median_mse_prediction_array, 
            median_mse_input_vs_gt_array, 
            percentile_mse_prediction_array, 
            percentile_mse_input_vs_gt_array,
            sum_chem_field_prediction_array,
            sum_chem_field_input_vs_gt_array,
            sum_cell_mask_prediction_array,
            sum_cell_mask_input_vs_gt_array,
            chem_cell_compare_prediction_array,
            chem_cell_compare_input_vs_gt_array,
            cell_fraction_prediction_array,
            cell_fraction_input_vs_gt_array,
            sum_vegf_field_prediction_array,
            sum_vegf_field_input_vs_gt_array,
            sum_cell_seg_prediction_array,
            sum_cell_seg_input_vs_gt_array,
            difference_sum_vegf_field_prediction,
            difference_sum_vegf_field_input_vs_gt,
            difference_sum_cell_seg_prediction,
            difference_sum_cell_seg_input_vs_gt,
            timestep_array
        )

def calculate_stats(df_list, column):
    '''Calculate the mean and standard deviation of the given column in the list of dataframes'''
    #concatenate all dataframes in the list
    df = pd.concat(df_list)

    #group by the index (timestep) and calculate mean and std
    mean = df.groupby('timestep')[column].mean()
    std = df.groupby('timestep')[column].std() 
    return mean, std

def load_and_calculate_stats(files, column1, column2):
    '''Load the data from the list of files and calculate the mean and standard deviation of the given columns'''
    data = [pd.read_csv(f, index_col='timestep') for f in files]
    mean1, std1 = calculate_stats(data, column1)
    mean2, std2 = calculate_stats(data, column2)
    return mean1, std1, mean2, std2

def augment_return_areas(single_image, plot=False):
    '''
    Augment the image and return the unique areas of the unique domains
    unique domains are defined as domains with unique areas AND unique shapes (by inertia tensor eigenvalues)

    Args:
    single_image (np.array): binary mask of the domain
    plot (bool): whether to plot the image and centroids and histogram of the unique areas

    Returns:
    unique_areas (np.array): unique domain areas
    '''

    # Create an empty array of the required size for augmentation
    augmented_image = np.zeros((3 * single_image.shape[0], 3 * single_image.shape[1]), dtype=np.float32)

    # Fill the center of the new array with the original image
    center_start = single_image.shape[0]
    center_end = 2 * single_image.shape[0]
    augmented_image[center_start:center_end, center_start:center_end] = single_image

    # Pattern the surrounding areas
    # Top and bottom
    augmented_image[:center_start, center_start:center_end] = single_image
    augmented_image[center_end:, center_start:center_end] = single_image

    # Left and right
    augmented_image[center_start:center_end, :center_start] = single_image
    augmented_image[center_start:center_end, center_end:] = single_image

    # corners
    augmented_image[:center_start, :center_start] = single_image
    augmented_image[:center_start, center_end:] = single_image
    augmented_image[center_end:, :center_start] = single_image
    augmented_image[center_end:, center_end:] = single_image


    flipped_augmented = 1-augmented_image


    aug_labels = label(flipped_augmented)
    aug_props = regionprops(aug_labels)
    aug_areas = [prop.area for prop in aug_props]
    
    aug_areas_inertia = [(prop.area, tuple(prop.inertia_tensor_eigvals)) for prop in aug_props]
    
    unique_regions_inertia = list(set(aug_areas_inertia))
    unique_areas = np.array([region[0] for region in unique_regions_inertia])
    unique_areas = unique_areas[unique_areas>3]


    if plot: #plots augmented with centroids, original next to original with centroids, and histogram of unique areas for debugging
        #centroids that correspond to the areas in unique_dropped_4
        fig,ax = plt.subplots()
        ax.imshow(flipped_augmented)
        centroids = []

        # Loop through each property in aug_props
        for prop in aug_props:
            # If the area of the property is in unique_dropped_4
            if prop.area in unique_areas:
                # Append the centroid of the property to the list
                centroids.append(prop.centroid)
        ax.scatter(*zip(*[(c[1], c[0]) for c in centroids]), color='red', s=5)
        ax.set_title('Augmented Image with Centroids labeled')
        ax.axis('off')
        plt.show()

        fig,ax = plt.subplots(1,2)
        flipped_original = 1-single_image
        ax[0].imshow(flipped_original)
        ax[1].imshow(flipped_original)
        original_labels = label(flipped_original)
        original_props = regionprops(original_labels)
        centroids_original = []

        # Loop through each property in original_props
        for prop in original_props:
            # If the area of the property is in unique_areas
            if prop.area in unique_areas:
                # Append the centroid of the property to the list
                centroids_original.append(prop.centroid)
        ax[1].scatter(*zip(*[(c[1], c[0]) for c in centroids_original]), color='red', s=5)
        ax[1].set_title('Original Image Centroids labeled')
        ax[1].axis('off')

        ax[0].set_title('Original Image')
        ax[0].axis('off')
        plt.show()

        # Plot the histogram of the areas
        plt.hist(unique_areas, bins='auto')
        plt.xlabel('Domain areas')
        plt.ylabel('Frequency')
        plt.title('Histogram of Unique Domain Areas')
        plt.show()

    return unique_areas # return the unique areas as a numpy array

def domain_area_periodic_histogram(prediction, ground_truth, null_hypothesis,  bins_option='auto', plot=False, distribution_distance=False, PDF=True):
    '''
    labels domains and calculates the area of each domain in the prediction and ground truth
    then plots a histogram of the domain areas
    
    Args:
    prediction (np.array): binary mask of the predicted domain
    ground_truth (np.array): binary mask of the ground truth domain
    null_hypothesis (np.array): binary mask of the null hypothesis domain
    bins_option (int): number of bins for the histogram, 'auto' for automatic binning using the numpy default Freedman-Diaconis method
    plot (bool): whether to plot the histogram
    distribution_distance (bool): whether to calculate the Wasserstein distance between the histograms of domain areas for the prediction vs ground truth and prediction vs null hypothesis
    PDF (bool): whether to plot the histogram as a probability density function
    
    Returns:
    pred_areas (list): list of areas of the domains in the prediction
    gt_areas (list): list of areas of the domains in the ground truth
    nh_areas (list): list of areas of the domains in the null hypothesis
    bins (np.array): consistent bin edges for plotting the area distributions as histograms
    if distribution_distance is True:
        emd_pred_gt (float): Wasserstein distance between the histograms of domain areas for the prediction vs ground truth
        emd_nh_gt (float): Wasserstein distance between the histograms of domain areas for the prediction vs null hypothesis
    '''

    #get the areas
    pred_areas = augment_return_areas(prediction)
    gt_areas = augment_return_areas(ground_truth)
    nh_areas = augment_return_areas(null_hypothesis)

    # Calculate the bin edges for all data combined for consistent bins using the bins option
    all_data = np.concatenate([pred_areas, gt_areas, nh_areas])
    _, bins = np.histogram(all_data, bins=bins_option)

    if plot:
        #plot the histogram
        fig, ax = plt.subplots()
        ax.hist(pred_areas, bins=bins, alpha=0.5, label='Prediction')
        ax.hist(gt_areas, bins=bins, alpha=0.5, label='Ground Truth')
        ax.hist(nh_areas, bins=bins, alpha=0.5, label='Null Hypothesis')
        ax.set_xlabel('Area')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.show()

    if distribution_distance:
        # Calculate the individual histograms with these bin edges, density=True normalizes the histograms to get a probability density
        pred_hist, pred_bins = np.histogram(pred_areas, bins=bins, density=PDF) #prediction
        gt_hist, gt_bins = np.histogram(gt_areas, bins=bins, density=PDF) #ground truth
        nh_hist, nh_bins = np.histogram(nh_areas, bins=bins, density=PDF) #null hypothesis

        # Calculate the Earth Mover's Distance (Waterstein Distance) between the histograms
        emd_pred_gt = wasserstein_distance(pred_bins[:-1], gt_bins[:-1], pred_hist, gt_hist) #prediction vs ground truth
        emd_nh_gt = wasserstein_distance(nh_bins[:-1], gt_bins[:-1], nh_hist, gt_hist) #null hypothesis vs ground truth

        return pred_areas, gt_areas, nh_areas, bins, emd_pred_gt, emd_nh_gt
    else:
        return pred_areas, gt_areas, nh_areas, bins

def evaluate_area_width_mean_median(timestep, evaluations, model, zipstore_path, timestep_ahead=100, bins_option='auto', plot=False, print_iter=False, hist_PDF=True):
    '''
    calculates mean and median domain areas and vessel widths for multiple evaluations
    
    Args:
    timestep (int): the timestep to start the evaluation
    evaluations (int): the number of iterations to evaluate the model
    model (torch model): the model to evaluate
    zipstore_path (str): the path to the zipstore containing the data
    timestep_ahead (int): the number of timesteps ahead that the model was trained to predict
    bins_option (int): number of bins for the histogram, 'auto' for automatic binning using the numpy default Freedman-Diaconis method
    plot (bool): whether to plot the EMD values
    hist_PDF (bool): whether to calculate the histograms as a probability density function

    Returns:
    timestep_array (list): list of timesteps evaluated
    emd_pred_gt_array (list): list of EMD values between the prediction and ground truth histograms
    emd_nh_gt_array (list): list of EMD values between the null hypothesis and ground truth histograms
    '''

    font_size = 20
    input_stack = None
    initial_fgbg, initial_vegf, initial_fgbg_next, initial_vegf_next = load_data(zipstore_path, timestep, timestep_ahead=timestep_ahead)
    
    timestep_array = []

    mean_domain_areas_pred = []
    mean_domain_areas_gt = []
    mean_vessel_widths_pred = []
    mean_vessel_widths_gt = []

    for i in range(evaluations):
        if print_iter:
            print("iteration ", i+1, "of ", evaluations)
        # Load data
        fgbg, vegf, fgbg_next, vegf_next = load_data(zipstore_path, timestep + i*timestep_ahead, timestep_ahead=timestep_ahead)
        if i == 0:
            input_stack = np.stack([fgbg, vegf], axis=-1)
            timestep_array.append(timestep + timestep_ahead) 
        else:
            input_stack = np.stack([output[:,:,0], output[:,:,1]], axis=-1)  # Use the previous output as input
            timestep_array.append(timestep + i*timestep_ahead + timestep_ahead) 
        if input_stack is None:
            raise ValueError("input_stack is None")
        
        # Pass through the model
        output = pass_through_model(model, input_stack)

        # Calculate the histogram statistics
        pred_areas, gt_areas, _, _, _, _ = domain_area_periodic_histogram(output[:,:,0], fgbg_next, initial_fgbg, bins_option=bins_option, plot=False, distribution_distance=True, PDF=hist_PDF)
        mean_domain_areas_pred.append(np.mean(pred_areas))
        mean_domain_areas_gt.append(np.mean(gt_areas))

        pred_widths, gt_widths, _, _, _, _ = width_distribution_histograms(output[:,:,0], fgbg_next, initial_fgbg, bins_option=bins_option, plot=False, distribution_distance=True, PDF=hist_PDF)
        mean_vessel_widths_pred.append(np.mean(pred_widths))
        mean_vessel_widths_gt.append(np.mean(gt_widths))

    return timestep_array, mean_domain_areas_pred, mean_domain_areas_gt, mean_vessel_widths_pred, mean_vessel_widths_gt

def evaluate_area_histogram_stats(timestep, evaluations, model, zipstore_path, timestep_ahead=100, bins_option='auto', plot=False, print_iter=False, hist_PDF=True):
    '''
    calculates the distance between the histogram of domain areas for multiple evaluations, accounting for periodic boundary conditions of the simulation
    
    Args:
    timestep (int): the timestep to start the evaluation
    evaluations (int): the number of iterations to evaluate the model
    model (torch model): the model to evaluate
    zipstore_path (str): the path to the zipstore containing the data
    timestep_ahead (int): the number of timesteps ahead that the model was trained to predict
    bins_option (int): number of bins for the histogram, 'auto' for automatic binning using the numpy default Freedman-Diaconis method
    plot (bool): whether to plot the EMD values
    hist_PDF (bool): whether to calculate the histograms as a probability density function

    Returns:
    timestep_array (list): list of timesteps evaluated
    emd_pred_gt_array (list): list of EMD values between the prediction and ground truth histograms
    emd_nh_gt_array (list): list of EMD values between the null hypothesis and ground truth histograms
    '''

    font_size = 20
    input_stack = None
    initial_fgbg, initial_vegf, initial_fgbg_next, initial_vegf_next = load_data(zipstore_path, timestep, timestep_ahead=timestep_ahead)
    
    timestep_array = []
    
    emd_pred_gt_array = []
    emd_nh_gt_array = []

    for i in range(evaluations):
        if print_iter:
            print("iteration ", i+1, "of ", evaluations)
        # Load data
        fgbg, vegf, fgbg_next, vegf_next = load_data(zipstore_path, timestep + i*timestep_ahead, timestep_ahead=timestep_ahead)
        if i == 0:
            input_stack = np.stack([fgbg, vegf], axis=-1)
            timestep_array.append(timestep + timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        else:
            input_stack = np.stack([output[:,:,0], output[:,:,1]], axis=-1)  # Use the previous output as input
            timestep_array.append(timestep + i*timestep_ahead + timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        if input_stack is None:
            raise ValueError("input_stack is None")
        
        # Pass through the model
        output = pass_through_model(model, input_stack)

        # Calculate the histogram statistics
        pred_areas_hist, gt_areas_hist, ho_hist, bins, emd_pred_gt, emd_nh_gt = domain_area_periodic_histogram(output[:,:,0], fgbg_next, initial_fgbg, bins_option=bins_option, plot=False, distribution_distance=True, PDF=hist_PDF)
        emd_pred_gt_array.append(emd_pred_gt)
        emd_nh_gt_array.append(emd_nh_gt)

    if plot:
        # Plot the EMD values on the same plot
        fig, ax = plt.subplots()
        ax.plot(timestep_array, emd_pred_gt_array, label='Prediction vs Ground Truth')
        ax.plot(timestep_array, emd_nh_gt_array, label='Null Hypothesis vs Ground Truth')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('EMD')
        ax.legend()
        ax.set_title('EMD between the histograms of domain areas')
        
        plt.show()
    return timestep_array, emd_pred_gt_array, emd_nh_gt_array

def evaluate_width_histogram_stats(timestep, evaluations, model, zipstore_path, timestep_ahead=100, bins_option='auto', plot=False, print_iter=False, hist_PDF=True):
    '''
    calculates the distance between the histogram of vessel width areas for multiple evaluations
    
    Args:
    timestep (int): the timestep to start the evaluation
    evaluations (int): the number of iterations to evaluate the model
    model (torch model): the model to evaluate
    zipstore_path (str): the path to the zipstore containing the data
    timestep_ahead (int): the number of timesteps ahead that the model was trained to predict
    bins_option (int): number of bins for the histogram, 'auto' for automatic binning using the numpy default Freedman-Diaconis method
    plot (bool): whether to plot the EMD values
    hist_PDF (bool): whether to calculate the histograms as a probability density function

    Returns:
    timestep_array (list): list of timesteps evaluated
    emd_pred_gt_array (list): list of EMD values between the prediction and ground truth histograms
    emd_nh_gt_array (list): list of EMD values between the null hypothesis and ground truth histograms
    '''

    font_size = 20
    input_stack = None
    initial_fgbg, initial_vegf, initial_fgbg_next, initial_vegf_next = load_data(zipstore_path, timestep, timestep_ahead=timestep_ahead)
    
    timestep_array = []
    
    emd_pred_gt_array = []
    emd_nh_gt_array = []

    for i in range(evaluations):
        if print_iter:
            print("iteration ", i+1, "of ", evaluations)
        # Load data
        fgbg, vegf, fgbg_next, vegf_next = load_data(zipstore_path, timestep + i*timestep_ahead, timestep_ahead=timestep_ahead)
        if i == 0:
            input_stack = np.stack([fgbg, vegf], axis=-1)
            timestep_array.append(timestep + timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        else:
            input_stack = np.stack([output[:,:,0], output[:,:,1]], axis=-1)  # Use the previous output as input
            timestep_array.append(timestep + i*timestep_ahead + timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        if input_stack is None:
            raise ValueError("input_stack is None")
        
        # Pass through the model
        output = pass_through_model(model, input_stack)

        # Calculate the histogram statistics
        pred_areas_hist, gt_areas_hist, ho_hist, bins, emd_pred_gt, emd_nh_gt = width_distribution_histograms(output[:,:,0], fgbg_next, initial_fgbg, bins_option=bins_option, plot=False, distribution_distance=True, PDF=hist_PDF)
        emd_pred_gt_array.append(emd_pred_gt)
        emd_nh_gt_array.append(emd_nh_gt)

    if plot:
        # Plot the EMD values on the same plot
        fig, ax = plt.subplots()
        ax.plot(timestep_array, emd_pred_gt_array, label='Prediction vs Ground Truth')
        ax.plot(timestep_array, emd_nh_gt_array, label='Null Hypothesis vs Ground Truth')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('EMD')
        ax.legend()
        ax.set_title('EMD between the histograms of domain areas')
        
        plt.show()
    return timestep_array, emd_pred_gt_array, emd_nh_gt_array

def width_distributions(single_image, plot_hist=False):
    '''
    calculates the local widths of the skeleton of the image and plots a histogram of the local widths

    Args:
    single_image (np.array): binary mask of the domain
    plot_hist (bool): whether to plot the histogram

    Returns:
    nonzero_widths (list): list of local widths of the skeleton
    '''

    # Compute the medial axis of the image 
    skel, distance =  medial_axis(single_image, return_distance=True)
    # Multiply the distance image by the skel image to keep only the distances (local width) on the skeleton
    dist_on_skel = distance * skel
    local_widths = dist_on_skel.ravel()
    nonzero_widths = local_widths[np.nonzero(local_widths)]

    if plot_hist:
        # Plot the histogram of the local widths with auto bins
        plt.hist(nonzero_widths, bins='auto')
        plt.xlabel('Local Width')
        plt.ylabel('Frequency')
        plt.title('Histogram of Local Widths')
        plt.show()
    
    return nonzero_widths

def width_distribution_histograms(prediction, ground_truth, null_hypothesis,  bins_option='auto', plot=False, distribution_distance=False, PDF=True):
    '''
    calculates distribution of local widths for prediction, ground truth and null hypothesis and plots histograms
    
    Args:
    prediction (np.array): binary mask of the predicted domain
    ground_truth (np.array): binary mask of the ground truth domain
    null_hypothesis (np.array): binary mask of the null hypothesis domain
    bins_option (int): number of bins for the histogram, 'auto' for automatic binning using the numpy default Freedman-Diaconis method
    plot (bool): whether to plot the histogram
    distribution_distance (bool): whether to calculate the Wasserstein distance between the histograms of domain areas for the prediction vs ground truth and prediction vs null hypothesis
    
    Returns:
    pred_areas (list): list of areas of the domains in the prediction
    gt_areas (list): list of areas of the domains in the ground truth
    nh_areas (list): list of areas of the domains in the null hypothesis
    bins (np.array): consistent bin edges for plotting the area distributions as histograms
    if distribution_distance is True:
        emd_pred_gt (float): Wasserstein distance between the histograms of domain areas for the prediction vs ground truth
        emd_nh_gt (float): Wasserstein distance between the histograms of domain areas for the prediction vs null hypothesis
    '''

    #get the areas
    pred_widths = width_distributions(prediction)
    gt_widths = width_distributions(ground_truth)
    nh_widths = width_distributions(null_hypothesis)

    # Calculate the bin edges for all data combined for consistent bins using the bins option
    all_data = np.concatenate([pred_widths, gt_widths, nh_widths])
    _, bins = np.histogram(all_data, bins=bins_option)

    if plot:
        #plot the histogram
        fig, ax = plt.subplots()
        ax.hist(pred_widths, bins=bins, alpha=0.5, label='Prediction')
        ax.hist(gt_widths, bins=bins, alpha=0.5, label='Ground Truth')
        ax.hist(nh_widths, bins=bins, alpha=0.5, label='Null Hypothesis')
        ax.set_xlabel('Area')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.show()

    if distribution_distance:
        # Calculate the individual histograms with these bin edges, density=True normalizes the histograms to get a probability density
        pred_hist, pred_bins = np.histogram(pred_widths, bins=bins, density=PDF) #prediction
        gt_hist, gt_bins = np.histogram(gt_widths, bins=bins, density=PDF) #ground truth
        nh_hist, nh_bins = np.histogram(nh_widths, bins=bins, density=PDF) #null hypothesis

        # Calculate the Earth Mover's Distance (Waterstein Distance) between the histograms
        emd_pred_gt = wasserstein_distance(pred_bins[:-1], gt_bins[:-1], pred_hist, gt_hist) #prediction vs ground truth
        emd_nh_gt = wasserstein_distance(nh_bins[:-1], gt_bins[:-1], nh_hist, gt_hist) #null hypothesis vs ground truth

        return pred_widths, gt_widths, nh_widths, bins, emd_pred_gt, emd_nh_gt
    else:
        return pred_widths, gt_widths, nh_widths, bins

#%% functions for performing evaluation on validation dataset
def load_model(checkpoint_path, num_loss_components, split_periodic=False, recursive=False, device='cuda'):
    '''
    loads a trained model from a checkpoint file

    Args:
    checkpoint_path (str): path to the checkpoint file
    num_loss_components (int): number of loss components in the model
    split_periodic (bool): whether the model is a split periodic model
    '''
    if recursive:
        return recursive_split_periodic_2_loss.load_from_checkpoint(checkpoint_path, map_location=torch.device(device))
    
    if split_periodic:
        return split_periodic_2_loss.load_from_checkpoint(checkpoint_path, map_location=torch.device(device))
    
    elif num_loss_components == 2:
        return periodic_unet_2_loss.load_from_checkpoint(checkpoint_path, map_location=torch.device(device))
    
    elif num_loss_components == 3:
        return periodic_unet_3_loss.load_from_checkpoint(checkpoint_path, map_location=torch.device(device))
    
    elif num_loss_components == 4:
        return periodic_unet_4_loss.load_from_checkpoint(checkpoint_path, map_location=torch.device(device))

def get_data_list(directory, exclude_list=None, file_extension='.zarr.zip'):
    '''
    returns a list of files in a directory with a given extension
    Args:
    directory (str): directory to search for files
    exclude_list (list): list of files to exclude
    file_extension (str): file extension to search for
    '''
    if exclude_list is None:
        exclude_list = []
    return [file for file in os.listdir(directory) if file.endswith(file_extension) and file not in exclude_list]

def save_dataframe(data_dict, folder_path, file_name):
    '''
    saves a dictionary of data as a csv file
    Args:
    data_dict (dict): dictionary of data to save
    folder_path (str): folder to save the file in
    file_name (str): name of the file to save
    '''
    df = pd.DataFrame(data_dict)
    file_path = os.path.join(folder_path, file_name)
    df.to_csv(file_path, index=False)

def evaluate_and_save_metrics(model, data_directory, file, validation_simulation_number, folder_paths, bins_option='auto', evaluate_domain_hist=False, timestep_start=2000, num_evaluations=100, timestep_ahead=100, hist_PDF=True):
    """
    Evaluates a model's performance on a given dataset and saves the evaluation metrics to CSV files.

    Args:
    model (object): loaded torch lightning model
    data_directory (str): The directory where the data files for validation are located.
    file (str): The name of the data file to be used for evaluation.
    validation_simulation_number (int): The number of the simulation run used for validation.
    folder_paths (dict): A dictionary where the keys are the names of the metrics and the values are the paths to the folders where the CSV files for each metric will be saved.
    bins_option (int, optional): method of calculating bins for the histogram. Defaults to 'auto' for automatic binning using the numpy default Freedman-Diaconis method.
    timestep_start (int, optional): The timestep to start the evaluation. Defaults to 2000.
    num_evaluations (int, optional): The number of iterations to evaluate the model. Defaults to 100.
    evaluate_domain_hist (bool, optional): If True, the function will also evaluate histogram stats. Defaults to False.
    timestep_ahead (int, optional): The number of timesteps ahead for the model to predict. Defaults to 100.
    hist_PDF (bool, optional): Whether to calculate the histograms as a probability density function. Defaults to True.

    Returns:
    None
    """
    if evaluate_domain_hist:
        (
            dice_coeff_array, 
            dice_coeff_original_array,
            avg_mse_prediction_array, 
            avg_mse_input_vs_gt_array,
            median_se_prediction_array,
            median_se_input_vs_gt_array,
            percentile_se_prediction_array,
            percentile_se_input_vs_gt_array,
            sum_chem_field_prediction_array,
            sum_chem_field_input_vs_gt_array,
            sum_cell_mask_prediction_array,
            sum_cell_mask_input_vs_gt_array,
            chem_cell_compare_prediction_array,
            chem_cell_compare_input_vs_gt_array,
            cell_fraction_prediction_array,
            cell_fraction_input_vs_gt_array,
            sum_vegf_field_prediction_array,
            sum_vegf_field_input_vs_gt_array,
            sum_cell_seg_prediction_array,
            sum_cell_seg_input_vs_gt_array,
            difference_sum_vegf_field_prediction,
            difference_sum_vegf_field_input_vs_gt,
            difference_sum_cell_seg_prediction,
            difference_sum_cell_seg_input_vs_gt,
            timestep_array,
            area_emd_pred_gt_array,
            area_emd_nh_gt_array,
            mean_se_areas_pred_gt_array,
            mean_se_areas_nh_gt_array,
            median_se_areas_pred_gt_array,
            median_se_areas_nh_gt_array,
            width_emd_pred_gt_array,
            width_emd_nh_gt_array,
            mean_se_width_pred_gt_array,
            mean_se_width_nh_gt_array,
            median_se_width_pred_gt_array,
            median_se_width_nh_gt_array
        ) = evaluate_model_with_hist(timestep_start, num_evaluations, model, os.path.join(data_directory, file), bins_option=bins_option, hist_PDF=hist_PDF, timestep_ahead=100, stats_percentile=75,plot=False, print_iter=False)
    else:
        (
            dice_coeff_array,
            dice_coeff_original_array,
            avg_mse_prediction_array,
            avg_mse_input_vs_gt_array,
            median_se_prediction_array,
            median_se_input_vs_gt_array,
            percentile_se_prediction_array,
            percentile_se_input_vs_gt_array,
            sum_chem_field_prediction_array,
            sum_chem_field_input_vs_gt_array,
            sum_cell_mask_prediction_array,
            sum_cell_mask_input_vs_gt_array,
            chem_cell_compare_prediction_array,
            chem_cell_compare_input_vs_gt_array,
            cell_fraction_prediction_array,
            cell_fraction_input_vs_gt_array,
            sum_vegf_field_prediction_array,
            sum_vegf_field_input_vs_gt_array,
            sum_cell_seg_prediction_array,
            sum_cell_seg_input_vs_gt_array,
            difference_sum_vegf_field_prediction,
            difference_sum_vegf_field_input_vs_gt,
            difference_sum_cell_seg_prediction,
            difference_sum_cell_seg_input_vs_gt,
            timestep_array
        ) = evaluate_model(timestep_start, num_evaluations, model, os.path.join(data_directory, file), timestep_ahead=timestep_ahead)

    metrics_data = [
        {
            'folder': folder_paths['dice'],
            'file_name': f'dice_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'dice_coeff_array': dice_coeff_array,
                'dice_coeff_original_array': dice_coeff_original_array
            }
        },
        {
            'folder': folder_paths['mse'],
            'file_name': f'mse_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'avg_mse_prediction_array': avg_mse_prediction_array,
                'avg_mse_input_vs_gt_array': avg_mse_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['median_se'],
            'file_name': f'median_se_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'median_mse_prediction_array': median_se_prediction_array,
                'median_mse_input_vs_gt_array': median_se_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['percentile_se'],
            'file_name': f'percentile_se_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'percentile_mse_prediction_array': percentile_se_prediction_array,
                'percentile_mse_input_vs_gt_array': percentile_se_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['sum_chem_se'],
            'file_name': f'sum_chem_se_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'sum_chem_field_prediction_array': sum_chem_field_prediction_array,
                'sum_chem_field_input_vs_gt_array': sum_chem_field_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['sum_cell_mask'],
            'file_name': f'sum_cell_mask_se_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'sum_cell_mask_prediction_array': sum_cell_mask_prediction_array,
                'sum_cell_mask_input_vs_gt_array': sum_cell_mask_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['chem_cell_compare'],
            'file_name': f'chem_cell_compare_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'chem_cell_compare_prediction_array': chem_cell_compare_prediction_array,
                'chem_cell_compare_input_vs_gt_array': chem_cell_compare_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['cell_fraction'],
            'file_name': f'cell_fraction_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'cell_fraction_prediction_array': cell_fraction_prediction_array,
                'cell_fraction_input_vs_gt_array': cell_fraction_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['sum_vegf_field'],
            'file_name': f'sum_vegf_field_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'sum_vegf_field_prediction_array': sum_vegf_field_prediction_array,
                'sum_vegf_field_input_vs_gt_array': sum_vegf_field_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['sum_cell_seg'],
            'file_name': f'sum_cell_seg_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'sum_cell_seg_prediction_array': sum_cell_seg_prediction_array,
                'sum_cell_seg_input_vs_gt_array': sum_cell_seg_input_vs_gt_array
            }
        },
        {
            'folder': folder_paths['difference_sum_vegf_field'],
            'file_name': f'difference_sum_vegf_field_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'difference_sum_vegf_field_prediction': difference_sum_vegf_field_prediction,
                'difference_sum_vegf_field_input_vs_gt': difference_sum_vegf_field_input_vs_gt
            }
        },
        {
            'folder': folder_paths['difference_sum_cell_seg'],
            'file_name': f'difference_sum_cell_seg_calculation_simulation_{validation_simulation_number}.csv',
            'data': {
                'timestep': timestep_array,
                'difference_sum_cell_seg_prediction': difference_sum_cell_seg_prediction,
                'difference_sum_cell_seg_input_vs_gt': difference_sum_cell_seg_input_vs_gt
            }
        }
    ]

    if evaluate_domain_hist:
        metrics_data.extend([
            {
                'folder': folder_paths['area_emd'],
                'file_name': f'area_emd_{validation_simulation_number}.csv',
                'data': {
                    'timestep': timestep_array,
                    'emd_pred_gt': area_emd_pred_gt_array,
                    'emd_nh_gt': area_emd_nh_gt_array
                }
            },
            {
                'folder': folder_paths['area_mean_se'],
                'file_name': f'area_mean_se_{validation_simulation_number}.csv',
                'data': {
                    'timestep': timestep_array,
                    'mean_se_areas_pred_gt': mean_se_areas_pred_gt_array,
                    'mean_se_areas_nh_gt': mean_se_areas_nh_gt_array
                }
            },
            {
                'folder': folder_paths['area_median_se'],
                'file_name': f'area_median_se_{validation_simulation_number}.csv',
                'data': {
                    'timestep': timestep_array,
                    'median_se_areas_pred_gt': median_se_areas_pred_gt_array,
                    'median_se_areas_nh_gt': median_se_areas_nh_gt_array
                }
            },
            {
                'folder': folder_paths['width_emd'],
                'file_name': f'width_emd_{validation_simulation_number}.csv',
                'data': {
                    'timestep': timestep_array,
                    'emd_pred_gt': width_emd_pred_gt_array,
                    'emd_nh_gt': width_emd_nh_gt_array
                }
            },
            {
                'folder': folder_paths['width_mean_se'],
                'file_name': f'width_mean_se_{validation_simulation_number}.csv',
                'data': {
                    'timestep': timestep_array,
                    'mean_se_width_pred_gt': mean_se_width_pred_gt_array,
                    'mean_se_width_nh_gt': mean_se_width_nh_gt_array
                }
            },
            {
                'folder': folder_paths['width_median_se'],
                'file_name': f'width_median_se_{validation_simulation_number}.csv',
                'data': {
                    'timestep': timestep_array,
                    'median_se_width_pred_gt': median_se_width_pred_gt_array,
                    'median_se_width_nh_gt': median_se_width_nh_gt_array
                }
            }
        ])

    for metric in metrics_data:
        save_dataframe(metric['data'], metric['folder'], metric['file_name'])

    # if evaluate_domain_hist:
    #     print('evaluating histogram stats')
    #     timestep_array, emd_pred_gt_array, emd_nh_gt_array = evaluate_area_histogram_stats(timestep_start, num_evaluations, model, os.path.join(data_directory, file), timestep_ahead=100, plot=False, print_iter=False, hist_PDF=hist_PDF)
    #     save_dataframe({
    #         'timestep': timestep_array,
    #         'emd_pred_gt': emd_pred_gt_array,
    #         'emd_nh_gt': emd_nh_gt_array
    #     }, folder_paths['area_emd'], f'area_emd_{validation_simulation_number}.csv')

    #     timestep_array, emd_pred_gt_array, emd_nh_gt_array = evaluate_width_histogram_stats(timestep_start, num_evaluations, model, os.path.join(data_directory, file), timestep_ahead=100, plot=False, print_iter=False, hist_PDF=hist_PDF)
    #     save_dataframe({
    #         'timestep': timestep_array,
    #         'emd_pred_gt': emd_pred_gt_array,
    #         'emd_nh_gt': emd_nh_gt_array
    #     }, folder_paths['width_emd'], f'width_emd_{validation_simulation_number}.csv')

def setup_directories(stats_data_directory, make_directories=True):
    '''
    creates directories for saving the evaluation metrics

    Args:
    stats_data_directory (str): the directory to save the evaluation metrics
    make_directories (bool): whether to create the directories if they do not exist. Keep this false if reading data thats already generated.

    Returns:
    directories (dict): a dictionary where the keys are the names of the metrics and the values are the paths to the folders where the CSV files for each metric will be saved    
    '''
    directories = {
        'dice': os.path.join(stats_data_directory, 'dice_data'),
        'mse': os.path.join(stats_data_directory, 'mse_data'),
        'median_se': os.path.join(stats_data_directory, 'median_mse_data'),
        'percentile_se': os.path.join(stats_data_directory, 'percentile_mse_data'),
        'sum_chem_se': os.path.join(stats_data_directory, 'sum_chem_se_data'),
        'sum_cell_mask': os.path.join(stats_data_directory, 'sum_mask_se_data'),
        'chem_cell_compare': os.path.join(stats_data_directory, 'chem_cell_compare_se_data'),
        'cell_fraction': os.path.join(stats_data_directory, 'frac_cells_se_data'),
        'sum_vegf_field': os.path.join(stats_data_directory, 'sum_vegf_field_data'),
        'sum_cell_seg': os.path.join(stats_data_directory, 'sum_cell_seg_data'),
        'area_emd': os.path.join(stats_data_directory, 'area_EMD_data'),
        'area_mean_se': os.path.join(stats_data_directory, 'area_mean_se_data'),
        'area_median_se': os.path.join(stats_data_directory, 'area_median_se_data'),
        'width_emd': os.path.join(stats_data_directory, 'width_EMD_data'),
        'width_mean_se': os.path.join(stats_data_directory, 'width_mean_se_data'),
        'width_median_se': os.path.join(stats_data_directory, 'width_median_se_data'),
        'difference_sum_vegf_field': os.path.join(stats_data_directory, 'difference_sum_vegf_field_data'),
        'difference_sum_cell_seg': os.path.join(stats_data_directory, 'difference_sum_cell_seg_data')
    }
    
    if make_directories:
        for folder in directories.values():
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    return directories

def load_csv_files_from_folder(folder):
    """
    Load all CSV files from a given folder.

    Parameters:
    - folder (str): The folder to load CSV files from.

    Returns:
    - list: List of file paths to CSV files in the folder.
    """
    return [os.path.join(os.getcwd(), folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

def load_all_metrics(folder_paths, TryDomainHist=False):
    """
    Load all CSV files and calculate statistics for each metric.

    Parameters:
    - folder_paths (dict): Dictionary of folder paths for each metric. Generated by setup_directories.
    - TryDomainHist (bool): Flag to determine if domain histogram metrics should be calculated. Histogram metrics for both widths and areas are calculated if True.

    Returns:
    - dict: Dictionary containing the calculated metrics. 
        Each key is paired with a tuple containing model_prediction_mean, model_prediction_std, h0_mean, h0_std
    """
    metrics = {}

    dice_files = load_csv_files_from_folder(folder_paths['dice'])
    mse_files = load_csv_files_from_folder(folder_paths['mse'])
    median_se_files = load_csv_files_from_folder(folder_paths['median_se'])
    percentile_se_files = load_csv_files_from_folder(folder_paths['percentile_se'])
    sum_chem_se_files = load_csv_files_from_folder(folder_paths['sum_chem_se'])
    sum_mask_se_files = load_csv_files_from_folder(folder_paths['sum_cell_mask'])
    chem_cell_compare_se_files = load_csv_files_from_folder(folder_paths['chem_cell_compare'])
    frac_cells_se_files = load_csv_files_from_folder(folder_paths['cell_fraction'])
    sum_vegf_field_files = load_csv_files_from_folder(folder_paths['sum_vegf_field'])
    sum_cell_seg_files = load_csv_files_from_folder(folder_paths['sum_cell_seg'])
    difference_sum_vegf_field_files = load_csv_files_from_folder(folder_paths['difference_sum_vegf_field'])
    difference_sum_cell_seg_files = load_csv_files_from_folder(folder_paths['difference_sum_cell_seg'])

    metrics['dice'] = load_and_calculate_stats(dice_files, 'dice_coeff_array', 'dice_coeff_original_array')
    metrics['mse'] = load_and_calculate_stats(mse_files, 'avg_mse_prediction_array', 'avg_mse_input_vs_gt_array')
    metrics['median_se'] = load_and_calculate_stats(median_se_files, 'median_mse_prediction_array', 'median_mse_input_vs_gt_array')
    metrics['percentile_se'] = load_and_calculate_stats(percentile_se_files, 'percentile_mse_prediction_array', 'percentile_mse_input_vs_gt_array')
    metrics['sum_chem_se'] = load_and_calculate_stats(sum_chem_se_files, 'sum_chem_field_prediction_array', 'sum_chem_field_input_vs_gt_array')
    metrics['sum_mask_se'] = load_and_calculate_stats(sum_mask_se_files, 'sum_cell_mask_prediction_array', 'sum_cell_mask_input_vs_gt_array')
    metrics['chem_cell_compare_se'] = load_and_calculate_stats(chem_cell_compare_se_files, 'chem_cell_compare_prediction_array', 'chem_cell_compare_input_vs_gt_array')
    metrics['frac_cells_se'] = load_and_calculate_stats(frac_cells_se_files, 'cell_fraction_prediction_array', 'cell_fraction_input_vs_gt_array')
    metrics['sum_vegf_field'] = load_and_calculate_stats(sum_vegf_field_files, 'sum_vegf_field_prediction_array', 'sum_vegf_field_input_vs_gt_array')
    metrics['sum_cell_seg'] = load_and_calculate_stats(sum_cell_seg_files, 'sum_cell_seg_prediction_array', 'sum_cell_seg_input_vs_gt_array')
    metrics['difference_sum_vegf'] = load_and_calculate_stats(difference_sum_vegf_field_files, 'difference_sum_vegf_field_prediction', 'difference_sum_vegf_field_input_vs_gt')
    metrics['difference_sum_cell_seg'] = load_and_calculate_stats(difference_sum_cell_seg_files, 'difference_sum_cell_seg_prediction', 'difference_sum_cell_seg_input_vs_gt')

    if TryDomainHist:
        area_emd_files = load_csv_files_from_folder(folder_paths['area_emd'])
        metrics['area_emd'] = load_and_calculate_stats(area_emd_files, 'emd_pred_gt', 'emd_nh_gt')
        area_mean_se_files = load_csv_files_from_folder(folder_paths['area_mean_se'])
        metrics['area_mean_se'] = load_and_calculate_stats(area_mean_se_files, 'mean_se_areas_pred_gt', 'mean_se_areas_nh_gt')
        area_median_se_files = load_csv_files_from_folder(folder_paths['area_median_se'])
        metrics['area_median_se'] = load_and_calculate_stats(area_median_se_files, 'median_se_areas_pred_gt', 'median_se_areas_nh_gt')
        width_emd_files = load_csv_files_from_folder(folder_paths['width_emd'])
        metrics['width_emd'] = load_and_calculate_stats(width_emd_files, 'emd_pred_gt', 'emd_nh_gt')
        width_mean_se_files = load_csv_files_from_folder(folder_paths['width_mean_se'])
        metrics['width_mean_se'] = load_and_calculate_stats(width_mean_se_files, 'mean_se_width_pred_gt', 'mean_se_width_nh_gt')
        width_median_se_files = load_csv_files_from_folder(folder_paths['width_median_se'])
        metrics['width_median_se'] = load_and_calculate_stats(width_median_se_files, 'median_se_width_pred_gt', 'median_se_width_nh_gt')

    return metrics

def plot_single_metric(metric, metric_name, ylabel, xlabel, model_legend_str='Surrogate Prediction', h0_label_str='Initial Condition', save_path=None, short_timescale=False, semilogy=False, title_str = True, title_size=20, label_size=15, tick_size=15, legend_size=10, prediction_color_str='blue', h0_color_str='orange'):
    """
    Plot a single metric.

    Parameters:
    - metric (tuple): Tuple containing model_prediction_mean, model_prediction_std, h0_mean, h0_std from load_and_calculate_stats or saved as a value in the metrics dictionary from load_all_metrics.
    - metric_name (str): Name of the metric.
    - title (str): Title of the plot.
    - ylabel (str): Label for the y-axis.
    - xlabel (str): Label for the x-axis.
    - model_legend_str (str): Legend label for the first data series (model prediction).
    - h0_label_str (str): Legend label for the second timeseries (initial condition).
    - save_path (str): Path to save the plot.
    - short_timescale (bool or list): if False, plot the entire timescale. If List, it is treated as the xlim
    - semilogy (bool): Flag to determine if the y-axis should be plotted on a semilog scale.
    - title_size (int): Font size for the title.
    - label_size (int): Font size for the labels.
    - tick_size (int): Font size for the ticks.
    - legend_size (int): Font size for the legend.

    Returns:
    - tuple: Tuple containing the x and y values for the model prediction and initial condition.
    """
    #unpack the metric tuple
    model_prediction_mean, model_prediction_std, h0_mean, h0_std = metric

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.plot(model_prediction_mean.index, model_prediction_mean, label=model_legend_str, color=prediction_color_str)
    ax.fill_between(model_prediction_mean.index, model_prediction_mean - model_prediction_std, model_prediction_mean + model_prediction_std, alpha=0.3, color=prediction_color_str)

    ax.plot(h0_mean.index, h0_mean, label=h0_label_str, color=h0_color_str)
    ax.fill_between(h0_mean.index, h0_mean - h0_std, h0_mean + h0_std, alpha=0.3, color=h0_color_str)

    if title_str and title_str == True: # set title as metric name if title_str == True
        ax.set_title(metric_name, fontsize=title_size)
    elif title_str: # set title as title_str if title_str is not True
        ax.set_title(title_str, fontsize=title_size)
    else: # no title
        pass

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size)

    # Set tick parameters
    ax.tick_params(axis='both', which='both', top=False, right=False, labelsize=tick_size)

    # Set x-axis limits if short_timescale_flag is True
    if short_timescale and short_timescale[0] is not None and short_timescale[1] is not None:
        #set xlim to specified range
        ax.set_xlim(model_prediction_mean.index[short_timescale[0]], model_prediction_mean.index[short_timescale[1]])
        #set ylim to the range of the data in the specified range
        ax.set_ylim(min(min(model_prediction_mean[short_timescale[0]:short_timescale[1]]), min(h0_mean[short_timescale[0]:short_timescale[1]])), max(max(model_prediction_mean[short_timescale[0]:short_timescale[1]]), max(h0_mean[short_timescale[0]:short_timescale[1]])))

    # Set y-axis to semilog scale if semilog_y_flag is True
    if semilogy:
        ax.set_yscale('log')

    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend_size:
        ax.legend(frameon=False, fontsize=legend_size)

    if save_path:
        plt.savefig(save_path)

    plt.show()
    if short_timescale:
        return model_prediction_mean[short_timescale[0]:short_timescale[1]], h0_mean[short_timescale[0]:short_timescale[1]], model_prediction_std[short_timescale[0]:short_timescale[1]], h0_std[short_timescale[0]:short_timescale[1]]
    else:
        return model_prediction_mean, h0_mean, model_prediction_std, h0_std

def plot_all_metrics_individual_plots(metrics, short_timescale=False, semilogy=False, title_size=20, label_size=15, tick_size=15):
    '''
    plots all metrics, pass in metrics dictionary from load_all_metrics() function.
    plots as individual plots in series

    Args:
    metrics (dict): dictionary of metrics
    short_timescale (bool or list): if False, plot the entire timescale. If List, it is treated as the xlim
    semilogy (bool): Flag to determine if the y-axis should be plotted on a semilog scale.
    title_size (int): Font size for the title.
    label_size (int): Font size for the labels.
    tick_size (int): Font size for the ticks.

    Returns:
    None
    '''

    for metric_name, metric in metrics.items():
        prediction, h0 = plot_single_metric(metric, metric_name, ylabel='Value', xlabel='Timestep', short_timescale=short_timescale, semilogy=semilogy, title_size=title_size, label_size=label_size, tick_size=tick_size)

def plot_all_metrics_subplots(metrics, short_timescale=False, semilogy=False, title_size=20, label_size=15, tick_size=15):
    '''
    Plots all metrics as subplots in a tiled layout with automatic text scaling.

    Args:
    metrics (dict): dictionary of metrics
    short_timescale (bool or list): if False, plot the entire timescale. If List, it is treated as the xlim
    semilogy (bool): Flag to determine if the y-axis should be plotted on a semilog scale.
    title_size (int): Base font size for the title.
    label_size (int): Base font size for the labels.
    tick_size (int): Base font size for the ticks.

    Returns:
    None
    '''
    
    num_metrics = len(metrics)
    cols = 2  # Number of columns for the subplot grid
    rows = int(np.ceil(num_metrics / cols))  # Calculate the number of rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Scale text size according to the number of subplots
    scale_factor = max(0.5, 1 - 0.05 * (rows * cols - 1))  # Adjust this scaling formula as needed
    scaled_title_size = title_size * scale_factor
    scaled_label_size = label_size * scale_factor
    scaled_tick_size = tick_size * scale_factor
    
    for i, (metric_name, metric) in enumerate(metrics.items()):
        ax = axes[i]
        model_prediction_mean, model_prediction_std, h0_mean, h0_std = metric

        ax.plot(model_prediction_mean.index, model_prediction_mean, label='Model Prediction')
        ax.fill_between(model_prediction_mean.index, model_prediction_mean - model_prediction_std, model_prediction_mean + model_prediction_std, alpha=0.3)

        ax.plot(h0_mean.index, h0_mean, label='Initial Condition')
        ax.fill_between(h0_mean.index, h0_mean - h0_std, h0_mean + h0_std, alpha=0.3)

        ax.set_title(metric_name, fontsize=scaled_title_size)
        ax.set_ylabel('Value', fontsize=scaled_label_size)
        ax.set_xlabel('Timestep', fontsize=scaled_label_size)

        # Set tick parameters
        ax.tick_params(axis='both', which='both', top=False, right=False, labelsize=scaled_tick_size)

        # Set x-axis limits if short_timescale is True
        if short_timescale and isinstance(short_timescale, list) and len(short_timescale) == 2:
            ax.set_xlim(model_prediction_mean.index[short_timescale[0]], model_prediction_mean.index[short_timescale[1]])

        # Set y-axis to semilog scale if semilogy is True
        if semilogy:
            ax.set_yscale('log')

        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)

    # Turn off empty axes if the number of metrics is odd
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()

def return_best_model(dir):
    '''
    loops through all folders and subfolders in the given directory
    returns the path of the .ckpt file that has the lowest validation loss first, 
    then highest epoch if there are multiple files with the same loss
    '''

    # Get a list of all .ckpt files in the directory
    ckpt_files = glob.glob(os.path.join(dir, '**', '*.ckpt'), recursive=True)
    
    # Parse the filenames to extract val_loss and epoch values
    parsed_files = []
    for file in ckpt_files:
        parts = os.path.basename(file).split('-')
        if len(parts) == 3:
            epoch_part, val_loss_part = parts[1], parts[2]
            epoch = float(epoch_part.split('=')[1])
            val_loss = float(val_loss_part.split('=')[1].split('.ckpt')[0])
            parsed_files.append((file, epoch, val_loss))
    
    # Sort the files first by val_loss (ascending), then by epoch (descending)
    parsed_files.sort(key=lambda x: (x[2], -x[1]))
    
    # Check that we found at least one .ckpt file
    if not parsed_files:
        raise ValueError(f"No valid .ckpt files found in directory {dir}")
    
    # Use the .ckpt file with the lowest val_loss and highest epoch
    model_checkpoint_path = parsed_files[0][0]

    return model_checkpoint_path

def evaluate_save_metrics_parallel(model_checkpoint_path, num_loss_components, split_periodic, recursive, data_directory, file, validation_simulation_number, folder_paths, evaluate_domain_hist, hist_PDF, bins_option):
    """
    Function to be executed in each process. Loads the model on the CPU and evaluates it.
    
    Parameters:
    - model_checkpoint_path (str): Path to the model checkpoint.
    - num_loss_components (int): The number of loss components in the model.
    - split_periodic (bool): Flag to determine if the model is a split periodic model.
    - recursive (bool): Flag to determine if the model is a recursive model.
    - data_directory (str): Directory where the data files are located.
    - file (str): The specific data file to be processed.
    - validation_simulation_number (str): The simulation number to be used.
    - folder_paths (dict): Dictionary of folder paths for saving metrics.
    - evaluate_domain_hist (bool): Flag to determine if domain histogram metrics should be calculated.
    - hist_PDF (bool): Flag to determine if histogram PDF should be calculated.

    Returns:
    - None
    """

    # Load the model on the CPU
    model = load_model(model_checkpoint_path, num_loss_components=num_loss_components, split_periodic=split_periodic, recursive=recursive, device='cpu')

    # Perform the evaluation and save metrics
    evaluate_and_save_metrics(model, data_directory, file, validation_simulation_number, folder_paths, bins_option=bins_option, evaluate_domain_hist=evaluate_domain_hist, hist_PDF=hist_PDF)

def run_stats_on_data(data_directory, stats_data_directory, model_checkpoint_path, num_loss_components,split_periodic=False, TryDomainHist=False, recursive=False, bins_option='auto',hist_PDF=True, parallel=False):
    """
    Run the evaluation metrics on the data in the data directory and save the results to the stats data directory.

    Parameters:
    - data_directory (str): The directory where the data files for validation are located.
    - stats_data_directory (str): The directory to save the evaluation metrics.
    - model_checkpoint_path (str): The path to the model checkpoint file.
    - num_loss_components (int): The number of loss components in the model.
    - split_periodic (bool): Flag to determine if the model is a split periodic model.
    - TryDomainHist (bool): Flag to determine if domain histogram metrics should be calculated.
    - recursive (bool): Flag to determine if the model is a recursive model.
    - hist_PDF (bool): Whether to calculate the histograms as a probability density function.
    - parallel (bool): Flag to determine if the evaluation should be run in parallel.

    Returns:
    - None
    """
    
    # Create necessary directories (this should be done before parallel execution)
    folder_paths = setup_directories(stats_data_directory)
    
    # Load data
    val_data_list = get_data_list(data_directory)

    final_validation_file_index = min(len(val_data_list), 100)

    if parallel:
        print('running in parallel')
        with ProcessPoolExecutor() as executor:
            futures = []
            for i, file in enumerate(val_data_list):
                if i >= final_validation_file_index:
                    print(f'reached {final_validation_file_index} files, stopping at {i} files')
                    break

                validation_simulation_number = file.split('_')[1].split('.')[0]
                print(f'validation_simulation_number = {validation_simulation_number}, file = {i}')

                # Submit the task to the executor
                futures.append(executor.submit(
                    evaluate_save_metrics_parallel,
                    model_checkpoint_path, num_loss_components, split_periodic, recursive,
                    data_directory, file, validation_simulation_number,
                    folder_paths, TryDomainHist, hist_PDF, bins_option
                ))

            # Wait for all futures to complete
            for future in futures:
                future.result()
    else:
        # Load model
        model = load_model(model_checkpoint_path, num_loss_components=num_loss_components, split_periodic=split_periodic, recursive=recursive)
        print('Model loaded')
        
        for i, file in enumerate(val_data_list):
            if i >= final_validation_file_index:
                print(f'reached {final_validation_file_index} files, stopping at {i} files')
                break

            validation_simulation_number = file.split('_')[1].split('.')[0]
            print(f'validation_simulation_number = {validation_simulation_number}, file = {i}')

            evaluate_and_save_metrics(model, data_directory, file, validation_simulation_number, folder_paths, evaluate_domain_hist=TryDomainHist, hist_PDF=hist_PDF, bins_option=bins_option)

    print('All files evaluated and saved')

def evaluate_model_return_images(timestep, evaluations, model, zipstore_path, timestep_ahead=100, plot=False, print_iter=False):
    '''
    Evaluate the model at a given timestep, propagates, and returns the images

    inputs:
    timestep: int, the timestep to start the evaluation
    evaluations: int, the number of iterations to evaluate the model
    model: torch model, the model to evaluate
    zipstore_path: str, the path to the zipstore containing the data
    timestep_ahead: int, the number of timesteps ahead that the model was trained to predict
    plot: bool, if True, plot the dice and error statistics
    print_iter: bool, if True, print the evaluation iteration number

    returns:
    ground_truth_fgbg: np.array, the ground truth cell mask
    ground_truth_vegf: np.array, the ground truth vegf field
    prediction_fgbg: np.array, the predicted cell mask
    prediction_vegf: np.array, the predicted vegf field
    timestep_array: list of ints, the timesteps evaluated
    '''
    
    font_size = 20
    input_stack = None
    # initial_fgbg, initial_vegf, initial_fgbg_next, initial_vegf_next = load_data(zipstore_path, timestep, timestep_ahead=timestep_ahead)
    

    for i in range(evaluations):
        if print_iter:
            print("iteration ", i+1, "of ", evaluations)
        # Load data
        fgbg, vegf, fgbg_next, vegf_next = load_data(zipstore_path, timestep + i*timestep_ahead, timestep_ahead=timestep_ahead)
        if i == 0:
            #initialize arrays
            ground_truth_fgbg = np.array([fgbg])
            ground_truth_vegf = np.array([vegf])
            prediction_fgbg = np.array([fgbg])
            prediction_vegf = np.array([vegf])
            timestep_array = [timestep]
            # create input stack for model
            input_stack = np.stack([fgbg, vegf], axis=-1)
            #append the initial values
        else:
            input_stack = np.stack([output[:,:,0], output[:,:,1]], axis=-1)  # Use the previous output as input

        timestep_array.append(timestep + i*timestep_ahead) #dice coefficients and mse are calculated for the _next  vs prediction
        # Pass through model
        output = pass_through_model(model, input_stack)
        #append the values
        prediction_fgbg = np.append(prediction_fgbg, [output[:,:,0]], axis=0)
        prediction_vegf = np.append(prediction_vegf, [output[:,:,1]], axis=0)
        ground_truth_fgbg = np.append(ground_truth_fgbg, [fgbg_next], axis=0)
        ground_truth_vegf = np.append(ground_truth_vegf, [vegf_next], axis=0)

    return ground_truth_fgbg, ground_truth_vegf, prediction_fgbg, prediction_vegf, timestep_array

def create_save_model_animation(model_name_str,movie_save_directory, model, zipstore_path, timestep=2000, evaluations=100, fps=4):
    '''
    Create and save movies for the cell and vegf predictions. The movies will be saved in the movie_save_directory.

    Args:
    model_name_str (str): The name of the model to be appended to the movie.
    movie_save_directory (str): The directory to save the movies.
    timestep (int): The timestep to start the evaluation.
    evaluations (int): The number of iterations to evaluate the model.
    model (object): The loaded torch lightning model.
    zipstore_path (str): The path to the zipped data file.

    Returns:
    None
    ''' 
    ground_truth_fgbg, ground_truth_vegf, prediction_fgbg, prediction_vegf, timestep_array = evaluate_model_return_images(timestep, evaluations, model, zipstore_path,timestep_ahead=100, plot=False, print_iter=False)
    assert ground_truth_fgbg.shape == prediction_fgbg.shape == ground_truth_vegf.shape == prediction_vegf.shape
    assert len(timestep_array) == ground_truth_fgbg.shape[0]

    # Use the 'dark_background' style
    plt.style.use('dark_background')

    title_size = 30
    suptitle_size = 35
    colorbar_label_size = 20

    # colorbar min and max values
    vmin = np.min(ground_truth_vegf)
    vmax = np.max(ground_truth_vegf)

    # Initialize the figures for the two movies
    fig_fgbg, axs_fgbg = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    fig_vegf, axs_vegf = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')

    # Function to update the top row (fgbg images)
    def update_fgbg(i):
        axs_fgbg[0].imshow(ground_truth_fgbg[i], cmap='gray')
        axs_fgbg[0].set_title(f'Simulation', fontsize=title_size)
        axs_fgbg[1].imshow(prediction_fgbg[i], cmap='gray')
        axs_fgbg[1].set_title(f'Model Prediction', fontsize=title_size)
        fig_fgbg.suptitle(f'Timestep {timestep_array[i]}', fontsize=suptitle_size)
        #axes off
        axs_fgbg[0].axis('off')
        axs_fgbg[1].axis('off')

    # Function to update the bottom row (vegf images)
    def update_vegf(i):
        axs_vegf[0].imshow(ground_truth_vegf[i], cmap='viridis', vmin=vmin, vmax=vmax)
        axs_vegf[0].set_title(f'Simulation', fontsize=title_size)
        axs_vegf[1].imshow(prediction_vegf[i], cmap='viridis', vmin=vmin, vmax=vmax)
        axs_vegf[1].set_title(f'Model Prediction', fontsize=title_size)

        #axes off
        axs_vegf[0].axis('off')
        axs_vegf[1].axis('off')

        # Add a colorbar to the bottom row
        divider = make_axes_locatable(axs_vegf[0])
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig_vegf.colorbar(axs_vegf[0].images[0], cax=cax, orientation='vertical', shrink=0.8)
        # cbar.set_label('Concentration', fontsize=title_size)
        cbar.ax.tick_params(labelsize=colorbar_label_size)
        cbar.outline.set_visible(False)
        fig_vegf.suptitle(f'Timestep {timestep_array[i]}', fontsize=suptitle_size)

    # Create animations
    anim_fgbg = FuncAnimation(fig_fgbg, update_fgbg, frames=len(timestep_array), repeat=False)
    anim_vegf = FuncAnimation(fig_vegf, update_vegf, frames=len(timestep_array), repeat=False)

    # Create the directory to save the movies
    os.makedirs(movie_save_directory, exist_ok=True)
    fgbg_movie_path = os.path.join(movie_save_directory, f'{model_name_str}_fgbg_movie.mp4')
    vegf_movie_path = os.path.join(movie_save_directory, f'{model_name_str}_vegf_movie.mp4')

    # Save the movies
    anim_fgbg.save(fgbg_movie_path, writer='ffmpeg', fps=fps)
    anim_vegf.save(vegf_movie_path, writer='ffmpeg', fps=fps)

    # plt.show()
    print('completed movies')

def run_save_mean_median_area_width(data_directory, stats_data_directory, model_checkpoint_path, num_loss_components,num_evaluations=100, timestep=2000, split_periodic=False, TryDomainHist=False, recursive=False, bins_option='auto',hist_PDF=True):
    """
    Run the evaluation metrics on the data in the data directory and save the results to the stats data directory.

    Parameters:
    - data_directory (str): The directory where the data files for validation are located.
    - stats_data_directory (str): The directory to save the evaluation metrics.
    - model_checkpoint_path (str): The path to the model checkpoint file.
    - num_loss_components (int): The number of loss components in the model.
    - split_periodic (bool): Flag to determine if the model is a split periodic model.
    - TryDomainHist (bool): Flag to determine if domain histogram metrics should be calculated.
    - recursive (bool): Flag to determine if the model is a recursive model.
    - hist_PDF (bool): Whether to calculate the histograms as a probability density function.

    Returns:
    - None
    """
    
    # Load data
    val_data_list = get_data_list(data_directory)

    final_validation_file_index = min(len(val_data_list), 100)

    # Load model
    model = load_model(model_checkpoint_path, num_loss_components=num_loss_components, split_periodic=split_periodic, recursive=recursive)
    print('Model loaded')
    
    for i, file in enumerate(val_data_list):
        if i >= final_validation_file_index:
            print(f'reached {final_validation_file_index} files, stopping at {i} files')
            break

        validation_simulation_number = file.split('_')[1].split('.')[0]
        print(f'validation_simulation_number = {validation_simulation_number}, file = {i}')
        zipstore_path = os.path.join(data_directory, file)
        timestep_array, mean_domain_areas_pred, mean_domain_areas_gt, mean_vessel_widths_pred, mean_vessel_widths_gt = evaluate_area_width_mean_median(timestep, num_evaluations, model, zipstore_path, timestep_ahead=100, bins_option='auto', plot=False, print_iter=False, hist_PDF=True)
        
        
    print('All files evaluated and saved')



# %%
