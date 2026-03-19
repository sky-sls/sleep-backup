import numpy as np
import os
import h5py
from ustaging.utils.scriptutils.extract import to_h5_file
from argparse import ArgumentParser

# ljy改 - 20221027，PAD上是2EEG
def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='EEGs.txt to h5py (PAD)')
    parser.add_argument("-f", type=str, required=True,
                        help='Path to file to read in.')
    parser.add_argument("-o", type=str, required=False,
                        help="Output path, default = ./filename.h5")
    # parser.add_argument("--logging_out_path", type=str, default=None,
    #                     help='Optional path to store prediction log. If not set, <out_folder>/<file_name>.log is used.')
    parser.add_argument("--channel_names", nargs='+', type=str, default='EEG1+EEG2', # "EEG1+EEG2"
                        required=False,
                        help="A list of channel name (CH1+CH2), e.g. 'EEG1+EEG2' ")
    parser.add_argument("--sample_rate", type=int, default=256,
                        help='sample_rate')
    return parser

# if __name__ == "__main__":
def run(args, dump_args=None):
    # 0. data & filename
    # Linux下的路径，形如：.../XXX.txt -- args.f, 如 D:/Codes/lab/D_paperPlot/h5py/txt/liyang.txt
    # out_path2 = f'D:/Codes/lab/D_paperPlot/h5py/dod_h/{subjectName}/{out_fname}.h5'
    # print(out_path2.split('/')[-1].split('.')[0])
    # out_path = f'D:/Codes/lab/D_paperPlot/h5py/dod_h/{subjectName}'
    data = None
    filename = 'signal'
    if args.f:
        # Set absolute input file path
        args.f = os.path.abspath(args.f)
        dir = os.path.split(args.f)[0]  # 分割最后一个'\', [0]=当前txt的目录
        filename_type = os.path.split(args.f)[-1]  # [-1]=XXX.txt

        data = np.loadtxt(args.f, delimiter=',')  # N * C
        filename = os.path.splitext(filename_type)[0] # 法1
        # 法2：filename = args.f.split('/')[-1].split('.')[0] # txt/liyang.txt
        print(f"read in: {filename}.txt")

        # Set output file path, 若args.o设为文件夹，则文件名保留为原来的txt文件名
        if args.o is None:
            args.o = os.path.join(dir, f'{filename}.h5')
        elif os.path.isdir(args.o):  # 若是file，则不处理
            args.o = os.path.join(args.o, os.path.splitext(os.path.split(args.f)[-1])[0] + ".h5")
        print(f"output path: {args.o}")



    # 1. out_path
    # subjectName = '1da3544e-dc5c-5795-adc3-f5068959211f'
    # out_fname = 'liyang_test'
    # out_path = os.path.join(f'D:/Codes/lab/D_paperPlot/h5py/dod_h/{subjectName}', f"{out_fname}.h5")
    out_path = args.o
    print(f"out_path: {out_path}")


    # 2. channel_names
    if args.channel_names:
        channel_names = args.channel_names.split('+')
    else:
        channel_names = ["EEG1", "EEG2"]

    # 3.其他
    sample_rate = args.sample_rate

    # data = np.loadtxt("D:/Codes/lab/D_paperPlot/h5py/txt/liyang.txt", delimiter=',')  # N*C
    to_h5_file(out_path, data, channel_names, sample_rate, None)


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args(args)
    run(args, dump_args=args)


if __name__ == "__main__":
    entry_func()



#
# def to_h5_file2(out_path, data, channel_names, sample_rate, date,
#                dtype=np.float32, compress=True, **kwargs):
#     """
#     Saves a NxC ndarray 'data' of PSG data (N samples, C channels) to a .h5
#     archive at path 'out_path'. A list 'channel_names' of length C must be
#     passed, giving the name of each channel in 'data'. Each Nx1 array in 'data'
#     will be stored under groups in the h5 archive according to the channel name
#
#     Also sets h5 attributes 'date' and 'sample_rate'.
#
#     Args:
#         out_path:      (string)   Path to a h5 archive to write to
#         data:          (ndarray)  A NxC shaped ndarray of PSG data
#         channel_names: (list)     A list of C strings giving channel names for
#                                   all channels in 'data'
#         sample_rate:   (int)      The sample rate of the signal in 'data'.
#         date:          (datetime) A datetime object. Is stored as a timetuple
#                                   within the archive. If a non datetime object
#                                   is passed, this will be stored 'as-is'.
#         dtype:         (np.dtype) The datatype to store the data as
#         **kwargs:
#     """
#     import h5py
#     import time
#     from datetime import datetime
#     if len(data.shape) != 2:
#         raise ValueError("Data must have exactly 2 dimensions, "
#                          "got shape {}".format(data.shape))
#     if data.shape[-1] == len(channel_names):
#         assert data.shape[0] != len(channel_names)  # Should not happen
#         data = data.T
#     elif data.shape[0] != len(channel_names):
#         raise ValueError("Found inconsistent data shape of {} with {} select "
#                          "channels ({})".format(data.shape,
#                                                 len(channel_names),
#                                                 channel_names))
#     if isinstance(date, datetime):
#         # Convert datetime object to TS naive unix time stamp
#         date = time.mktime(date.timetuple())
#     if isinstance(channel_names, ChannelMontageTuple):
#         channel_names = channel_names.original_names
#     data = data.astype(dtype)
#     with h5py.File(out_path, "w") as out_f:
#         out_f.create_group("channels")
#         for chan_dat, chan_name in zip(data, channel_names):
#             out_f['channels'].create_dataset(chan_name,
#                                              data=chan_dat,
#                                              chunks=True,
#                                              compression='gzip')
#         out_f.attrs['date'] = date or "UNKNOWN"
#         out_f.attrs["sample_rate"] = sample_rate
