'''
this script runs the evaluation statistics on a separate set of validation data from the training runs.
Uses utility functions defined in evaluate_stats_functions.py to run the statistics and save the results.

Note: code has options such as model_names, model_intermediate_names, and num_loss_components for generating data to compare different model architectures and training
techniques designed to improve upon the baseline model. The best model as described in the paper was periodic_weighted, bce_2loss_1_10 which corresponds to training a model with
bce and mse loss from Periodic_Unet_2_loss.py
'''

#%% imports
import sys
import evaluate_stats_functions as esf
import os
import time

#%% run stats on data
val_data_directory = r'' #directory where validation data is stored
stats_data_parant_directory ="" #directory where run_evaluation data will be saved
models_parent_directory = '' #directory where the trained models are stored
models_intermediate_names = ['periodic_split_weighted', 'periodic_weighted']
model_names = ['bce_2loss_1_10', 'dice_2loss_1_10'] # folder convention must be parent_directory/model_name/
num_loss_components = 2

def main(): 
    start_time = time.time()
    models_intermediate_names = ['periodic_split_weighted', 'periodic_weighted']
    bins_options = ['fd', 'sturges']
    pdf_options = [1, 0]
    for model_type in models_intermediate_names:

        if 'split' in model_type.lower():
            split_periodic = True
        else:
            split_periodic = False
        for model_name in model_names:
            for bin_option in bins_options:
                for pdf_option in pdf_options:
                    print('model name:', model_name)
                    print('model type:', model_type)
                    print('bin option:', bin_option)
                    print('pdf option:', pdf_option)
                    hist_pdf_str = 'hist_PDF_true' if pdf_option else 'hist_PDF_false'
                    stats_data_directory = os.path.join(stats_data_parant_directory, model_type, hist_pdf_str, bin_option, model_name)
                    #make directory if it does not exist
                    os.makedirs(stats_data_directory, exist_ok=True)
                    model_checkpoint = esf.return_best_model(os.path.join(models_parent_directory, model_type, model_name))
                    print(model_checkpoint)
                    esf.run_stats_on_data(val_data_directory, stats_data_directory, model_checkpoint, num_loss_components=num_loss_components,split_periodic=split_periodic, TryDomainHist=True, hist_PDF=pdf_option, parallel=False)

    end_time = time.time()
    print('time taken ', (end_time - start_time) / 60, ' minutes')

if __name__ == "__main__":
    main()

