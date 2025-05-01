'''
compares evaluation speed between 100 cc3d calls and 100 unet surrogate evaluations
'''
#%% imports
import os
import time
import torch
import numpy as np
import zarr
import evaluate_stats_functions as esf #utility functions file evaluate_stats_functions.py in Data_Analysis_Code folder
from os.path import dirname, join, expanduser
from cc3d.CompuCellSetup.CC3DCaller import CC3DCaller
import gc
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime


def call_cc3d_sim(sim_fname, output_path):
    cc3d_sim_folder = 'Angiogenesis_random/Angiogenesis3.cc3d' #NOTE TO ENSURE PATH IS CORRECT
    workspace_path = os.path.join(output_path, 'workspace_dump')
    os.makedirs(workspace_path, exist_ok=True)
    cc3d_caller = CC3DCaller(
        cc3d_sim_fname=sim_fname,
        output_dir = workspace_path)
    ret_value = cc3d_caller.run()
    del cc3d_caller
    gc.collect()
    return

def rewrite_xml(xml_file_path, new_pif_name):
    '''
    rewrites xml file with new pif name for loading in cc3d

    xml_file_path: str, path to xml file
    new_pif_name: str, new pif name to replace in xml file
    '''
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    pif_name_element = root.find('.//PIFName')

    if pif_name_element is not None:
        #replace with new string
        pif_name_element.text = new_pif_name

        tree.write(xml_file_path)
    else:
        print('<PIFName> element not found in the XML file.')

def load_and_stack_for_model_eval(zarr_file):
    '''Load the data from the zipstore at the given timestep index and at index + 10 for the 10 timesteps ahead'''
    with zarr.open(store=zarr.storage.ZipStore(zipstore_path, mode="r")) as root:
        fgbg = np.array(root["fgbg"][:])  
        vegf = np.array(root["vegf"][:])
        input_stack = np.stack([fgbg, vegf], axis=-1)
    return input_stack
    
def pass_through_model_time_on_device(model, input_stack,threshold=0.5,probabilities=False, do_warmup=False):
    '''Pass the input stack through the model and return the output, sigmoided and thresholded and time'''
    input_tensor = torch.from_numpy(input_stack).float().unsqueeze(0).permute(0, 3, 1, 2) #convert to tensor with shape (1, 2, 256, 256), (batch, channels, height, width)

    input_tensor = input_tensor.to(model.device)

    model.eval()

    with torch.no_grad():

        if do_warmup:
            _ = model(input_tensor)

        time_on_device_start = time.time()
        output = model(input_tensor) # output is [batch_size, 2, 256, 256]
        time_on_device_eval = time.time() - time_on_device_start #time taken for evaluation on device

    if probabilities:
        output[:,0,:,:] = torch.sigmoid(output[:,0,:,:]) #sigmoided for probabilities
    else:
        output[:,0,:,:] = torch.sigmoid(output[:,0,:,:])
        output[:,0,:,:] = (output[:,0,:,:]>threshold).int() #thresholded probabilities to binary mask

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy() #convert back to numpy with shape (256, 256, 2)
    return output, time_on_device_eval

evaluate_100mcs_folder = 'evaluate_100mcs_cc3d'
dir_up_for_workspace_dump = '' #CC3D must generate workspace files, create a folder for this and delete later.
zarr_folder = r'Angiogenesis_random_save_piff_100\evaluate_100mcs_cc3d\Simulation\zarr_files'
model_checkpoint_folder = r'' #trained model checkpoint path
model_checkpoint = esf.return_best_model(model_checkpoint_folder)
model = esf.load_model(model_checkpoint, 2, split_periodic=False)


sim_files_folder = os.path.join(evaluate_100mcs_folder, 'Simulation')
xml_file_path = os.path.join(sim_files_folder, 'Angiogenesis.xml')
cc3d_sim_path = os.path.join(evaluate_100mcs_folder, 'Angiogenesis3.cc3d')

#%% model evaluations zarr
evaluations_dict = {}

print('unet model evaluations')
unet_model_evaluation_times = []
for i, zarr_file in enumerate(os.listdir(zarr_folder)):
    if zarr_file.endswith('.zarr.zip'):
        if i == 0:
            do_warmup = True
        else:
            do_warmup = False
        zipstore_path = os.path.join(zarr_folder, zarr_file)
        input_stack = load_and_stack_for_model_eval(zipstore_path)
        output, eval_time = pass_through_model_time_on_device(model, input_stack, do_warmup=do_warmup)
        print('time taken for model evaluation:', eval_time)
        unet_model_evaluation_times.append(eval_time)

print('done with unet model evaluations')
print('mean time for unet model evaluations:', np.mean(unet_model_evaluation_times))
print('std time for unet model evaluations:', np.std(unet_model_evaluation_times))

evaluations_dict['unet_model_evaluations'] = unet_model_evaluation_times

#%% cc3d evaluations
print('cc3d evaluations')
cc3d_evaluation_times = []
for file in os.listdir(evaluate_100mcs_folder):
    if file.endswith('.piff'):
        piff_file = file
        rewrite_xml(xml_file_path, piff_file)
        start_time = time.time()
        call_cc3d_sim(cc3d_sim_path, dir_up_for_workspace_dump)
        end_time = time.time()
        cc3d_evaluation_times.append(end_time - start_time)

#%%
print('done with cc3d evaluations')
print('mean time for cc3d evaluations:', np.mean(cc3d_evaluation_times))
print('std time for cc3d evaluations:', np.std(cc3d_evaluation_times))

print('mean time for unet model evaluations:', np.mean(unet_model_evaluation_times))
print('std time for unet model evaluations:', np.std(unet_model_evaluation_times))

evaluations_dict['cc3d_evaluations'] = cc3d_evaluation_times

#save evaluations dict to csv with pandas
evaluations_df = pd.DataFrame(evaluations_dict)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
folder_with_timestamp = os.path.join(evaluate_100mcs_folder, f'evaluations_comparison_cc3d_surrogate_{timestamp}')
os.makedirs(folder_with_timestamp, exist_ok=True)
evaluations_df.to_csv(os.path.join(folder_with_timestamp, 'evaluations_comparison_cc3d_surrogate.csv'))
print('evaluations saved to csv')

# %% load plot data

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
data_path = 'evaluations_comparison_cc3d_surrogate.csv'
# Load the data
evaluations_df = pd.read_csv(data_path)

# Constants for plot appearance
title_size = 30
label_size = 35
tick_size = 30
bar_color = 'none'  # Change to any color you prefer
dot_color = ['darkturquoise','orange']    # Color of the datapoints
dot_edgecolors = 'black'  # Color of the dots' edges
dot_size = 50
error_color = 'black'  # Color of the error bars
alpha_datapoints = 0.4   # Transparency of the datapoints

# Exclude the index column
evaluations_df_no_index = evaluations_df.iloc[:, 1:]

#new column names
evaluations_df_no_index.columns = ['Surrogate', 'CC3D']

# Calculate mean and standard deviation (for error bars)
means = evaluations_df_no_index.mean()
standard_errors = evaluations_df_no_index.std()

# Create the figure for the bar chart, normal y axis
plt.figure(figsize=(8, 10),dpi=200)
lower_error = np.zeros_like(standard_errors)
upper_error = standard_errors
asymmetric_error = [lower_error, upper_error]

bars = plt.bar(
    evaluations_df_no_index.columns,
    means,
    yerr=asymmetric_error,
    capsize=5,
    color=bar_color,  # Fill color
    edgecolor='black',  # Border color
    linewidth=1.5,  # Thickness of the border
    error_kw={'elinewidth': 3.5, 'ecolor': error_color},
    width=0.5,
    zorder=10,
)

# Overlay individual data points with jitter and transparency
for i, col in enumerate(evaluations_df_no_index.columns):
    jittered_x = np.ones(len(evaluations_df_no_index[col])) * i + np.random.uniform(-0.1, 0.1, len(evaluations_df_no_index[col]))
    plt.scatter(
        jittered_x,
        evaluations_df_no_index[col],
        facecolors=dot_color[i],  # Fill color of the dots
        edgecolors=dot_edgecolors,    # Border color of the dots
        alpha=alpha_datapoints,
        zorder=5,
        s=dot_size
    )
plt.ylabel('Evaluation Time (s)', fontsize=label_size)
plt.xticks(fontsize=label_size)
plt.yticks(fontsize=tick_size)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yscale('log')

plt.show()
# %%
