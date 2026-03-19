# import h5py

# # H5 文件路径
# h5_file_path = 'your_file_path.h5'

# # 打开 H5 文件并检查其结构
# with h5py.File(h5_file_path, 'r') as f:
#     # 查看根目录的键值
#     keys = list(f.keys())
#     print(f"Keys in the H5 file: {keys}")
    
#     # 检查是否包含 'channels' 键
#     if 'channels' in keys:
#         # 查看 'channels' 中的所有通道
#         channels = list(f['channels'].keys())
#         print(f"Channels in the file: {channels}")
#     else:
#         print("No 'channels' group found in the H5 file.")

'''
sky h5read --file_regex 'datasets/processed_SKY/dcsm/*psg.h5' \
--out_dir 'datasets/processed_SKY/plots/'

sky h5read --file_regex 'datasets/dcsm/tp0a0acb70_3ad6_4d2e_ab5c_b89fc9f28011/*psg.h5' \
--out_dir 'datasets/processed_SKY/tryplots/' \
--channels 'ECG-II' 'ABDOMEN' 'THORAX' \
--rename_channels 'ECG' 'ABD' 'THO' \
--resample 128 --overwrite
'''


import h5py
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os

# 默认假设采样率为 256 Hz
DEFAULT_SAMPLING_RATE = 128

def get_argparser():
    """
    Returns an argument parser for this script.
    """
    parser = ArgumentParser(description='Read and plot signals from an H5 file.')
    
    # 接受命令行参数
    parser.add_argument("--file_regex", type=str, required=True,
                        help='Path pattern to the H5 files (supports * wildcard).')
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where the extracted files will be saved.")
    parser.add_argument("--channels", nargs="+", type=str, 
                        help="Space-separated list of channels to extract and plot. If not provided, all channels will be plotted.")
    parser.add_argument("--rename_channels", nargs="+", type=str,
                        help="Space-separated list of channel names to rename. Must match in length with --channels.")
    parser.add_argument("--resample", type=int, default=None,
                        help="Resample the signals to the specified rate.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files.")
    return parser

def read_and_plot_signals(h5_file, channels=None, renamed_channels=None, sampling_rate=DEFAULT_SAMPLING_RATE, out_dir=None):
    """
    Reads the signals from the H5 file and plots 5 minutes of data for each channel.
    If no channels are provided, all available channels are plotted.
    """
    with h5py.File(h5_file, 'r') as f:
        # Check available channels in the file
        available_channels = list(f['channels'].keys())
        print(f"Available channels: {available_channels}")
        
        # If no specific channels were provided, use all available channels
        if not channels:
            channels = available_channels
        
        # Extract each selected channel and plot
        for idx, channel in enumerate(channels):
            if channel not in available_channels:
                print(f"Warning: Channel '{channel}' not found in the file.")
                continue
            
            # Read the signal for the current channel
            signal_data = f['channels'][channel][:]
            
            # Get middle 5 minutes (start from the middle of the data)
            start_idx = len(signal_data) // 2 - sampling_rate * 1 * 60 // 2
            end_idx = start_idx + sampling_rate * 1 * 60
            
            # Get the 5-minute segment
            signal_segment = signal_data[start_idx:end_idx]
            
            # Plot the signal for this channel
            plt.figure(figsize=(15, 6))  # Create a new figure for each channel
            if renamed_channels and len(renamed_channels) == len(channels):
                renamed_channel = renamed_channels[idx]
            else:
                renamed_channel = channel  # Use original name if no rename
            
            time = np.linspace(0, 1*60, sampling_rate * 1 * 60)
            plt.plot(time, signal_segment)
            plt.title(f'{renamed_channel} - 5 minutes (Assumed {sampling_rate} Hz Sampling Rate)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Save the plot to the specified output directory
            if out_dir:
                plt.savefig(os.path.join(out_dir, f'{renamed_channel}_5min_plot.png'))
            plt.close()  # Close the figure to avoid overlapping plots

def run(args):
    # This function processes the file(s), extracts channels, and plots data
    files = [args.file_regex]  # You can expand this to handle multiple files with globbing
    
    # Check if output directory exists, if not create it
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # Ensure resampling rate is set to default if not provided
    sampling_rate = args.resample if args.resample is not None else DEFAULT_SAMPLING_RATE
    
    # Process each file
    for h5_file in files:
        print(f"Processing file: {h5_file}")
        read_and_plot_signals(h5_file, args.channels, args.rename_channels, sampling_rate, args.out_dir)

def entry_func(args=None):
    # Print a welcome message before starting
    print("Welcome to the H5 Signal Extraction and Plotting Script!")
    print("This script extracts specified channels from an H5 file and plots the first 5 minutes of data.")
    print("Ensure you provide the correct file path pattern and desired channels.")
    
    parser = get_argparser()
    run(parser.parse_args(args))

if __name__ == "__main__":
    entry_func()







