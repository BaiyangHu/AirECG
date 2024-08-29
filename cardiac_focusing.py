from scipy.signal import resample
import numpy as np
from scipy.signal import resample
from tslearn.metrics import dtw as tsl_dtw
from multiprocessing import Pool, shared_memory
import micro_motion_amp

def linear_time_warp(signal, target_length):
    """
    Linearly time warps the signal to the target length.
    """
    warped_signal = resample(signal, target_length)
    return warped_signal

def pattern_matching(signal):
    hmax = 200
    hmin = 100
    # TRY STEP = 2hmax
    #T_c = [signal[i:i+hmax] for i in range(0, len(signal), hmax)] # Generate corse template for cardiac motion window with length hmax
    T_c = T_c = [signal[0:hmax]]
    step = 2 # Define how the point step we calculates dtw

    # Step 1: Calculating the dtw scores for all segments
    dtw_index_scores = [] # Store each dtw value in terms of [index,score]

    index = 0 # Index of the starting point in the segment
    while index + hmax < len(signal): # Traverse all suitable starting index in the whole signal
        dtw = np.sum([tsl_dtw(signal[index:index+hmax], template) for template in T_c])
        dtw_index_scores.append([index,dtw])
        index += step
    dtw_index_scores = np.array(dtw_index_scores)

    # Step 2: Find the index of segment corresponds to the smallest dtw
    i = 0 # Starting index
    indexes = [] # Index list of segment with smallest dtw (overlapped)
    while True:
        search_range = dtw_index_scores[(dtw_index_scores[:,0] >= i + hmin) & (dtw_index_scores[:,0] <= i + hmax)]
        if search_range.shape[0] == 0:
            break

        i = search_range[np.argmin(search_range[:,1]), 0]
        indexes.append(i)
    
    # Step 3: Reform the non-overlapping segmentation
    S_dagger = [] # Non-overalpped signal segment
    T_dagger = np.zeros(hmax) # Final template

    for i in range(len(indexes)-1):
        S_dagger_i = signal[int(indexes[i]):int(indexes[i+1])]
        S_dagger.append(S_dagger_i) # Non-overlapped signal with various length
        T_dagger += linear_time_warp(S_dagger_i, hmax) # Linear time wrap signal length to hmax

    # Step 4: Update the template
    T_dagger /= len(indexes) # Average of non-overlapped signal uses as final template
    # print(T_dagger)
    # print(len(T_dagger))
    # return T_dagger

    # Step 5: Calculate the dtw score
    # Calculate the average score of the signal
    pattern_score = 0
    for dtws in S_dagger: # Traverse all segment, calculate the average dtw score
        pattern_score += tsl_dtw(dtws, T_dagger)
    pattern_score /= len(S_dagger)

    return T_dagger, pattern_score, indexes

def calculate_power(signal):
    """
    Calculate the power of a signal as the mean of the squared amplitudes.
    """
    return np.mean(np.square(signal))

def focusing(signal, thr=1500):
    points = []
    selected_voxels = []
    voxel_powers = []
    
    for x in range(9):
        for y in range(9):
            for z in range(17):
                # Apply second derivative to the unwrapped phase signal
                seq = micro_motion_amp.second_dev(np.unwrap(np.angle(signal[x,y,z,:])),1/200)
                
                # Perform pattern matching
                T_dagger, pattern_score, indexes = pattern_matching(seq)
                print(f'x: {x}, y: {y}, z: {z}, score: {pattern_score}')
                
                # Select the voxel if the pattern score is below the threshold
                if pattern_score <= thr:
                    points.append([x, y, z, pattern_score])  # Store voxel location and DTW score
                    selected_voxels.append(signal[x, y, z, :])  # Store the corresponding signal
                    
                    # Calculate the power of the signal
                    voxel_powers.append(calculate_power(signal[x, y, z, :]))

    # Convert the selected voxel positions and power into voxel_info format
    voxel_info = [(x, y, z, power) for (x, y, z, _), power in zip(points, voxel_powers)]
    
    # Convert selected_voxels into a 2D array for voxel_signals
    voxel_signals = np.array(selected_voxels)
    
    print(f'Selected Points: {points}')
    print(f'Number of selected voxels: {len(selected_voxels)}')
    
    return voxel_info, voxel_signals