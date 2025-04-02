

material_type = 'SOFC'  #'SOFC, SSB, CL
REMind_mode = 'SG2'  # 'SG1': In-plane Super Resolution, or 'SG2': Through-plane Reconstruction
img_size = 128
sf = 4      #super resolution factor #4, 8, 16. Notably, 8 was chosen for the paper.
lr=5e-5     #learning rate
num_epochs = 10000
load_pretrained = False #Allow to keep training the model starting with the given state.
load_pretrained_epoch = 1200 # 1200 for example
ema_decay = 0.9999
N_mean = 1

L=6 #number of slices in SG2, e.g., 4, 6, 8, 10, 12

input_channels = 3  # decided by the types of embeddings in DDPM
if REMind_mode == 'SG1':
    input_channels = 3
elif REMind_mode == 'SG2':
    input_channels = 5
#paths
train_dataset_path_SG1 = f'./d_data/Stage1_sf={sf}_1x{img_size}x{img_size}_tiff_train'
test_dataset_path_SG1  = f'./d_data/Stage1_sf={sf}_1x{img_size}x{img_size}_tiff_test'
train_dataset_path_SG2 = f'./d_data/Stage2_{L}x{img_size}x{img_size}_tiff_train'
test_dataset_path_SG2 = f'./d_data/Stage2_{L}x{img_size}x{img_size}_tiff_test'

sparse_FIB_SEM_low_res_stack_path = f'./d_data/sparse_FIB_SEM_low_res_stack'
monitor_process_path = './f_monitoring_processes'
checkpoint_path_SG1 = './d_data/checkpoints_SG1'
checkpoint_path_SG2 = './d_data/checkpoints_SG2'

#inference
check_point_SG1_inference_path = './d_data/checkpoints_SG1/model_2200.pth'
check_point_SG2_inference_path = './d_data/checkpoints_SG2/model_2200.pth'
infenrece_img_SG1_path = './d_data/SG1_inference'
infenrece_img_SG2_path = './d_data/SG2_inference'
sparse_FIB_SEM_high_res_stack_path = './d_data/sparse_FIB_SEM_high_res_stack'
refined_FIB_SEM_high_res_stack_path = './d_data/refined_FIB_SEM_high_res_stack'


#hyperparameters
num_time_steps = 5000
batch_size = 64
batch_size_inference = 2    #could be 200 if the test dataset include 200 samples
num_groups = 32