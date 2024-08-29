'''
This file contains code that implements the first data preprocessing stage in reproducing the result from the paper
3D Beamforming

It takes the .bin file output by the mmWave radar and generate a .pkl file after 3D beamforming
'''

import numpy as np
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

def calculation( 
        dataCube,
        voxel_location,
        idx_info,
        antenna_location,
        info,
        c = 299792458  # Speed of light
    ):
    '''
    Given the dataCube from beamforming, calculate the intensity of each voxel.

    return: The indexes of the voxel and its intensity
    '''
    s = np.zeros(info["clip"], dtype=np.complex64)
    for frame_idx in range(info["clip"]): # Traverse all the frames
        for channel in range(1, 12 + 1):
            r = np.linalg.norm(np.array(voxel_location) - np.array(antenna_location[channel])).astype(np.float64) * 2 # Calculating distance
            t = np.linspace(2e-7, 51.2e-6, num=256)
            phase_shift = np.exp(1j * 2 * np.pi * info["freqSlope"] * r * t / c) * np.exp(1j * 2 * np.pi * r / info["waveLength"])
            
            # Antennas 1、3、5、7、9、11 need to inverse the phase
            if channel in [1, 3, 5, 7, 9, 11]:
                phase_shift *= -1
            
            for chirp_idx in range(1):  # 1 chirp
                tx_idx = (channel - 1) // info["numRxAntennas"]
                rx_idx = (channel - 1) % info["numRxAntennas"]

                y_nt = dataCube[frame_idx, chirp_idx, tx_idx, rx_idx, :]

                s[frame_idx] += np.sum(y_nt * phase_shift) # Summing the intensity
    return [idx_info, s]

class preProcessing:
    def __init__(self, settings):
        # This name is used to save the .pkl file, based on our naming logic, the name should be the absolute time when the data is collected
        self.save_name = settings["save_name"]
        # The absolute time recorded in the .txt file of the radar data, which is the log file
        self.start_time = settings["start_time"]
        # This is calculated when loading the data, based on the length of the data, ADC samples and virtual antennas
        self.numFrames = settings["numFrames"]
        '''
        The parameters below are mainly decided by the hardware configuration
        '''
        # Usually 256 in our set up
        self.numADCSamples = settings["numADCSamples"]
        # In our radar set up - 3
        self.numTxAntennas = settings["numTxAntennas"]
        # In our radar set up - 4
        self.numRxAntennas = settings["numRxAntennas"]
        # Calculated the total transmission channels
        self.numVirtualAntennas = self.numTxAntennas * self.numRxAntennas
        # 2 Loops per frame
        self.numLoopsPerFrame = settings["numLoopsPerFrame"]
        # Starting frequency 60 GHz
        self.startFreq = settings["startFreq"]
        # Sampling rate 5000 ksps
        self.sampleRate = settings["sampleRate"]
        # Frquency slope 64.997e12 Hz/s
        self.freqSlope = settings["freqSlope"]
        # Idle time 10 us
        self.idleTime = settings["idleTime"]
        # Ramp end time 60 us
        self.rampEndTime = settings["rampEndTime"]
        # Calculation for number of chirps per frame
        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        # Size of angle of arrival using fft
        self.numAOA = settings["numAOA"]
        # Wavelength of the radar
        self.waveLength = 0.005 
        # By default, we process all clips (all frames)
        self.clip = settings["clip"]
        # Load the 3D Beamforming data
        self.dataCube = self.load_bin(settings["path"])
        # Printing progress bar
        self.progress = tqdm(total=9*17*9, desc="Beamforming")

    def load_bin(self, path):
        '''
        input : path for the .bin file

        output: data + parameters of radar raw data - [no.frame, no.chirps, Tx, Rx, no.ADCsample (complex number)]
        '''
        loaded = np.array([])
        loaded = np.concatenate([loaded, np.fromfile(path, dtype = np.int16)])
        # Calculate the number of frame based on the length of data and other parameters
        # 2*2 -> 2 for IQ data component and 2 loops
        self.numFrames = len(loaded)//(2 * 2 * self.numADCSamples * self.numVirtualAntennas)
        if self.clip == -1: # By default, we process all frames
            self.clip = self.numFrames
        # Shape the data so we can fit all frames
        loaded = loaded.reshape(self.numFrames, -1)
        # First reshape
        loaded = np.reshape(
            loaded,
            (
                -1, # No. of frames
                self.numLoopsPerFrame,
                self.numTxAntennas,
                self.numRxAntennas,
                self.numADCSamples // 2, # One set of IQ pais
                2, # IQ components
                2, # I part and Q part
            ),
        )
        # Change the order of the last two dimension so that we can combine them in the next step
        loaded = np.transpose(loaded, (0, 1, 2, 3, 4, 6, 5))
        # Reducing the last dimension and combine the IQ complex values
        loaded = np.reshape(
            loaded,
            (
                -1,
                self.numLoopsPerFrame,
                self.numTxAntennas,
                self.numRxAntennas,
                self.numADCSamples,
                2, # IQ components
            ),
        )
        loaded = (
            1j * loaded[:, :, :, :, :, 0] + loaded[:, :, :, :, :, 1]
        ).astype(np.complex64) # Imaginary I component and Real Q component
        loaded = loaded[:self.clip, :, :, :, :] # Takes only the first clips frames
        return loaded
    
    def beamforming(self):
        """
        Voxelisation of time-dimension signal
        
        Based on equatino given in the paper:
        S(x,y,z,t)=\sum_{n=1}^{N}\sum_{t=1}^{T}y_{n,t}e^{j2\pi\frac{kr(x,y,z,n)}{c}}e^{j2\pi\frac{r(x,y,z,n)}{\lambda}}

        Antennas configuration
        4  2
        3  1
        12 10 8 6
        11 9  7 5

        voxel size: 9*17*9
        range: x, y, z = [-40, 40], [35, 60], [-40, 40] cm
        """
        wavelength = self.waveLength
        antenna_loc = {  # Define the location of each antenna in the form of (x,y,z)
            4: (-wavelength/4, 0, 5*wavelength/4),    2: (wavelength/4, 0, 5*wavelength/4),
            3: (-wavelength/4, 0, 3*wavelength/4),    1: (wavelength/4, 0, 3*wavelength/4),  
            12:(-wavelength/4, 0, wavelength/4),      10:(wavelength/4, 0, wavelength/4),    8: (3*wavelength/4, 0, wavelength/4),      6: (5*wavelength/4, 0, wavelength/4),
            11:(-wavelength/4, 0, -wavelength/4),     9: (wavelength/4, 0, -wavelength/4),   7: (3*wavelength/4, 0, -wavelength/4),     5: (5*wavelength/4, 0, -wavelength/4),
        }
        voxel_dim = (9,9,17) # Define the number of voxels in (x,y,z) which corresponds to (height,depth,width)
        self.voxel_dim = voxel_dim
        voxel_range = { # Define the length of each dimension, x and z are fixed, y depends on the height of the radar
            "x": (-0.4,0.4),
            "y": (0.35,0.6), # if the torso is y metres away from the radar, then (y - 0.1, y + 0.15)
            "z": (-0.4,0.4),
        }
        self.voxel_range = voxel_range
        voxel_size = { # Calculate the size of each voxel
            "x": (voxel_range["x"][1] - voxel_range["x"][0])
            / (voxel_dim[0] - 1),
            "y": (voxel_range["y"][1] - voxel_range["y"][0])
            / (voxel_dim[1] - 1),
            "z": (voxel_range["z"][1] - voxel_range["z"][0])
            / (voxel_dim[2] - 1),
        }
        # Define the voxel map for 3D representation
        # Consider a 3D voxelisation model, with each voxel contains the data of all frame
        self.voxel_map = np.zeros((voxel_dim[0], voxel_dim[1], voxel_dim[2], self.clip), dtype=np.complex64)

        # Define the paramaters needed for 3D beamforming
        info = {
            "numFrames": self.numFrames,
            "numRxAntennas": self.numRxAntennas,
            "freqSlope": self.freqSlope,
            "waveLength": self.waveLength,
            "clip": self.clip,
        }
        print("Starting 3D beamforming now...")
        with Pool(processes=8) as pool:
            # Using DataCube from shared memory
            # shm = shared_memory.SharedMemory(name=self.shm_name)
            # shared_dataCube = np.ndarray(self.dataCube.shape, dtype=self.dataCube.dtype, buffer=shm.buf)
            for x_idx in range(voxel_dim[0]):
                for y_idx in range(voxel_dim[1]):
                    for z_idx in range(voxel_dim[2]):
                        voxel_location = ( # Voxel location in Cartesian plane
                            voxel_range["x"][0] + x_idx * voxel_size["x"],
                            voxel_range["y"][0] + y_idx * voxel_size["y"],
                            voxel_range["z"][0] + z_idx * voxel_size["z"],
                        )
                        idx_info = (x_idx, y_idx, z_idx) # Store the indexes of the Voxel
                        pool.apply_async( # Multiprocessing to speed up
                            calculation,
                            (self.dataCube, voxel_location, idx_info, antenna_loc, info),
                            callback=self.callback,
                        )
                        
            pool.close()
            pool.join()
        return self.voxel_map
    
    def callback(self, ret): # Call back function for multiprocessing
        self.voxel_map[ret[0][0], ret[0][1], ret[0][2], :] = ret[1]
        self.progress.update(1)

    def save(self):
        '''
        Save the 3D beamforming data as a .pkl file
        '''
        with open("./pkl/{}.pkl".format((self.save_name)), "wb") as f:
            pickle.dump(
                [self.voxel_map, {
                    "voxel_range" : self.voxel_range,
                    "clip" : self.clip,
                    "voxel_dim" : self.voxel_dim,
                    "start_time": self.start_time
                }]
                , f)
        print("Save successfully!")


if __name__ == "__main__":
    path = r"C:\Users\78406\Desktop\mmWave-Postprocessing\1722923241.8222_Raw_0.bin"
    log_path = r"C:\Users\78406\Desktop\mmWave-Postprocessing\1722923241.8222.txt"
    with open(log_path, "r") as f:
        log = f.readlines()
        start_time = log[1].split(" ")[-1]
    settings = {
        "path": path, # Path for the .bin file
        "save_name": path.split("\\")[-1].split(".")[0], # Save as this name
        "numFrames": 36000, # 没用 现在会计算
        "start_time": start_time,
        "clip" : -1, # 处理所有帧
        "numADCSamples": 256,
        "numTxAntennas": 3,
        "numRxAntennas": 4,
        "numLoopsPerFrame": 2,
        "startFreq": 60, # GHz
        "sampleRate": 5000, # ksps
        "freqSlope": 64.997e12, # Hz/s 注意单位
        "idleTime": 10, # us
        "rampEndTime": 60, # us
        "numAOA": 64,
    }
    processor = preProcessing(settings)
    processor.beamforming()
    processor.save()