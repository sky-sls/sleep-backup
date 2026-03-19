"""
Small utility script that extracts a set of channels from a set of PSG files
and saves them to a folder in .h5 files with minimally required header info
attached as h5 attributes (sample rate etc.).

The PSG file must be loadable using:
ustaging.io.high_level_file_loaders import load_psg
"""

from argparse import ArgumentParser
from glob import glob
import os
from ustaging.errors import ChannelNotFoundError
from ustaging.io.channels import ChannelMontageTuple, ChannelMontageCreator
from ustaging.io.header import extract_header
from mpunet.logging import Logger


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Extract a set of channels from a set '
                                        'of PSG files, various formats '
                                        'supported. The extracted data will be'
                                        ' saved to .h5 files with minimal '
                                        'header information attributes.')
    parser.add_argument("--file_regex", type=str,
                        help='A glob statement matching all files to extract '
                             'from')
    parser.add_argument("--out_dir", type=str,
                        help="Directory in which extracted files will be "
                             "stored")
    parser.add_argument("--channels", nargs="+", type=str,
                        help="Space-separated list of CHAN1-CHAN2 format of"
                             "referenced channel montages to extract. A "
                             "montage will be created if the referenced "
                             "channel is not already available in the file. If"
                             " the channel does not already exist and if "
                             "CHAN1 or CHAN2 is not available, an error is "
                             "raised.")
    parser.add_argument("--rename_channels", nargs="+", type=str,
                        help="Space-separated list of channel names to save"
                             " as instead of the originally extracted names. "
                             "Must match in length --channels.")
    parser.add_argument('--resample', type=int, default=None,
                        help='Re-sample the selected channels before storage.')
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files of identical name")
    parser.add_argument("--use_dir_names", action="store_true",
                        help='Each PSG file will be saved as '
                             '<parent directory>.h5 instead of <file_name>.h5')
    return parser


def filter_channels(renamed_channels, selected_original_channels,
                    original_channels):
    inds_selected = [i for i, chan in enumerate(original_channels)
                     if chan in selected_original_channels]
    return [chan for i, chan in enumerate(renamed_channels)
            if i in inds_selected]


#original # def _extract(file_,
#              out_path,
#              channels,
#              renamed_channels,
#              logger,
#              args):
#     from ustaging.io.high_level_file_loaders import load_psg
#     from ustaging.utils.scriptutils import to_h5_file
#     channels_in_file = extract_header(file_)["channel_names"]

#     chan_creator = ChannelMontageCreator(existing_channels=channels_in_file,
#                                          channels_required=channels,
#                                          allow_missing=True)
#     logger("[*] Channels in file: " + ", ".join(chan_creator.existing_channels.names))
#     logger("[*] Output channels: " + ", ".join(chan_creator.output_channels.names))
#     logger("[*] Channels to load: " + ", ".join(chan_creator.channels_to_load.names))

#     # # sky0304在重命名之前添加调试信息
#     # logger("[DEBUG] args.channels: {}".format(args.channels))
#     # logger("[DEBUG] renamed_channels: {}".format(renamed_channels))
#     # logger("[DEBUG] Original channel names: {}".format(header['channel_names'].original_names))
#     # logger("[DEBUG] Output channel names: {}".format(header['channel_names'].names))

    
#     try:
#         # psg, header = load_psg(file_, chan_creator.channels_to_load, # ljy改13 - 20210723
#         psg, header = load_psg(file_,
#                                load_channels=chan_creator.channels_to_load,
#                                check_num_channels=False)
#     except ChannelNotFoundError as e:
#         logger("\n-----\nCHANNEL ERROR ON FILE {}".format(file_))
#         logger(str(e) + "\n-----")
#         os.rmdir(os.path.split(out_path)[0])
#         return

#     # create montages
#     psg, channels = chan_creator.create_montages(psg)
#     header['channel_names'] = channels

#     # Resample
#     logger("[*] PSG shape before re-sampling: {}".format(psg.shape))
#     if args.resample:
#         from ustaging.preprocessing.psg_sampling import set_psg_sample_rate
#         psg = set_psg_sample_rate(psg,
#                                   new_sample_rate=args.resample,
#                                   old_sample_rate=header['sample_rate'])
#         header['sample_rate'] = args.resample
#         if psg.shape[0] % args.resample:
#             logger("ERROR: Not divisible by sample rate!")
#     logger("[*] PSG shape after re-sampling: {}".format(psg.shape))

#     # Rename channels
#     if renamed_channels:
#         org_names = header['channel_names'].original_names
#         header['channel_names'] = filter_channels(renamed_channels,
#                                                   org_names,
#                                                   args.channels)
#     else:
#         header['channel_names'] = header['channel_names'].original_names
#     logger("[*] Extracted {} channels: {}".format(psg.shape[1],
#                                                   header['channel_names']))
#     to_h5_file(out_path, psg, **header)

#sky0304修改
def _extract(file_,
             out_path,
             channels,
             renamed_channels,
             logger,
             args):
    from ustaging.io.high_level_file_loaders import load_psg
    from ustaging.utils.scriptutils import to_h5_file
    channels_in_file = extract_header(file_)["channel_names"]

    chan_creator = ChannelMontageCreator(existing_channels=channels_in_file,
                                         channels_required=channels,
                                         allow_missing=True)
    logger("[*] Channels in file: " + ", ".join(chan_creator.existing_channels.names))
    logger("[*] Output channels: " + ", ".join(chan_creator.output_channels.names))
    logger("[*] Channels to load: " + ", ".join(chan_creator.channels_to_load.names))
    try:
        # psg, header = load_psg(file_, chan_creator.channels_to_load, # ljy改13 - 20210723
        psg, header = load_psg(file_,
                               load_channels=chan_creator.channels_to_load,
                               check_num_channels=False)
    except ChannelNotFoundError as e:
        logger("\n-----\nCHANNEL ERROR ON FILE {}".format(file_))
        logger(str(e) + "\n-----")
        os.rmdir(os.path.split(out_path)[0])
        return

    # create montages
    psg, channels = chan_creator.create_montages(psg)
    header['channel_names'] = channels

    # Resample
    logger("[*] PSG shape before re-sampling: {}".format(psg.shape))
    if args.resample:
        from ustaging.preprocessing.psg_sampling import set_psg_sample_rate
        psg = set_psg_sample_rate(psg,
                                  new_sample_rate=args.resample,
                                  old_sample_rate=header['sample_rate'])
        header['sample_rate'] = args.resample
        if psg.shape[0] % args.resample:
            logger("ERROR: Not divisible by sample rate!")
    logger("[*] PSG shape after re-sampling: {}".format(psg.shape))

    # ========== 修正版：信号滤波处理 ==========
    from scipy import signal
    import numpy as np
    
    fs = header['sample_rate']  # 采样率
    logger(f"[*] Applying filters (fs={fs} Hz)")
    
    # 获取当前通道的原始名称（用于识别通道类型）
    current_channels = header['channel_names'].original_names
    
    # 对每个通道应用不同的滤波
    for i, chan_name in enumerate(current_channels):

        psg_col = psg[:, i].astype(np.float64)


        
        # ECG滤波：1-45Hz 带通（提高低端频率，避免数值不稳定）
        if chan_name.startswith('ECG') or chan_name == 'ECG-II':
            logger(f"[*] Applying ECG filter to channel {i}: {chan_name}")
            try:
                # 使用2阶滤波器，更稳定
                b, a = signal.butter(2, [1/(fs/2), 45/(fs/2)], 'band')
                psg_col = signal.filtfilt(b, a, psg_col)
                logger(f"[*] ECG filter applied successfully")
            except Exception as e:
                logger(f"[WARNING] ECG filter failed: {e}, using original")
        
        # 呼吸信号滤波：使用更稳定的方法
        elif (chan_name.startswith('ABDOMEN') or chan_name.startswith('THORAX') or 
              chan_name.startswith('ABD') or chan_name.startswith('THO')):
            logger(f"[*] Applying respiratory filter to channel {i}: {chan_name}")
            try:
                # 方法1：先去除趋势，再用低通滤波（更稳定）
                from scipy import signal
                
                # 去除线性趋势
                psg_detrended = signal.detrend(psg_col)
                
                # 使用2阶低通滤波，截止频率0.8Hz（保留呼吸信号）
                b, a = signal.butter(2, 0.8/(fs/2), 'low')
                psg_col = signal.filtfilt(b, a, psg_detrended)
                
                # 可选：再做一个很弱的高通去除极低频漂移（0.05Hz）
                # b_hp, a_hp = signal.butter(1, 0.05/(fs/2), 'high')
                # psg_col = signal.filtfilt(b_hp, a_hp, psg_col)
                
                logger(f"[*] Respiratory filter applied successfully")
            except Exception as e:
                logger(f"[WARNING] Respiratory filter failed: {e}, using original")
                psg_col = psg[:, i]  # 恢复原始信号
        
        # 其他信号：只做去趋势处理
        else:
            logger(f"[*] Applying detrend to channel {i}: {chan_name}")
            try:
                from scipy import signal
                psg_col = signal.detrend(psg_col)
            except:
                pass
        
        # 检查是否有NaN
        if np.any(np.isnan(psg_col)):
            logger(f"[WARNING] NaN detected in channel {i}, using original signal")
            psg_col = psg[:, i]
        
        # 更新数据
        psg[:, i] = psg_col
    
    logger("[*] Filtering completed")
    # ======================================

    # ========== 最终修正版 ==========
    # Rename channels - 使用原始的args.channels字符串列表
    if renamed_channels:
        # 获取原始的通道名称列表（从命令行传入的）
        original_channel_names = args.channels  # 这是原始的字符串列表
        
        # 创建映射
        channel_to_new = dict(zip(original_channel_names, renamed_channels))
        logger(f"[DEBUG] Mapping: {channel_to_new}")
        
        # 获取实际输出的通道名称
        output_channels = header['channel_names'].original_names
        logger(f"[DEBUG] Output channels: {output_channels}")
        
        # 按照原始顺序重命名
        renamed_list = []
        for orig_chan in original_channel_names:
            # 在输出通道中查找匹配
            found = False
            for out_chan in output_channels:
                # 处理后缀（如 -None）
                if out_chan.startswith(orig_chan):
                    renamed_list.append(channel_to_new[orig_chan])
                    found = True
                    logger(f"[DEBUG] Matched {orig_chan} -> {out_chan} -> {channel_to_new[orig_chan]}")
                    break
            if not found:
                logger(f"[WARNING] Channel {orig_chan} not found in output")
        
        # 更新通道名称
        header['channel_names'] = renamed_list
        logger(f"[*] Final channel names: {renamed_list}")
    else:
        header['channel_names'] = header['channel_names'].original_names
    # =================================

    logger("[*] Extracted {} channels: {}".format(psg.shape[1],
                                                  header['channel_names']))
    to_h5_file(out_path, psg, **header)



# without filtering
#def _extract(file_,
#              out_path,
#              channels,
#              renamed_channels,
#              logger,
#              args):
#     from ustaging.io.high_level_file_loaders import load_psg
#     from ustaging.utils.scriptutils import to_h5_file
#     channels_in_file = extract_header(file_)["channel_names"]

#     chan_creator = ChannelMontageCreator(existing_channels=channels_in_file,
#                                          channels_required=channels,
#                                          allow_missing=True)
#     logger("[*] Channels in file: " + ", ".join(chan_creator.existing_channels.names))
#     logger("[*] Output channels: " + ", ".join(chan_creator.output_channels.names))
#     logger("[*] Channels to load: " + ", ".join(chan_creator.channels_to_load.names))
#     try:
#         # psg, header = load_psg(file_, chan_creator.channels_to_load, # ljy改13 - 20210723
#         psg, header = load_psg(file_,
#                                load_channels=chan_creator.channels_to_load,
#                                check_num_channels=False)
#     except ChannelNotFoundError as e:
#         logger("\n-----\nCHANNEL ERROR ON FILE {}".format(file_))
#         logger(str(e) + "\n-----")
#         os.rmdir(os.path.split(out_path)[0])
#         return

#     # create montages
#     psg, channels = chan_creator.create_montages(psg)
#     header['channel_names'] = channels

#     # Resample
#     logger("[*] PSG shape before re-sampling: {}".format(psg.shape))
#     if args.resample:
#         from ustaging.preprocessing.psg_sampling import set_psg_sample_rate
#         psg = set_psg_sample_rate(psg,
#                                   new_sample_rate=args.resample,
#                                   old_sample_rate=header['sample_rate'])
#         header['sample_rate'] = args.resample
#         if psg.shape[0] % args.resample:
#             logger("ERROR: Not divisible by sample rate!")
#     logger("[*] PSG shape after re-sampling: {}".format(psg.shape))

#     # ========== 最终修正版 ==========
#     # Rename channels - 使用原始的args.channels字符串列表
#     if renamed_channels:
#         # 获取原始的通道名称列表（从命令行传入的）
#         original_channel_names = args.channels  # 这是原始的字符串列表
        
#         # 创建映射
#         channel_to_new = dict(zip(original_channel_names, renamed_channels))
#         logger(f"[DEBUG] Mapping: {channel_to_new}")
        
#         # 获取实际输出的通道名称
#         output_channels = header['channel_names'].original_names
#         logger(f"[DEBUG] Output channels: {output_channels}")
        
#         # 按照原始顺序重命名
#         renamed_list = []
#         for orig_chan in original_channel_names:
#             # 在输出通道中查找匹配
#             found = False
#             for out_chan in output_channels:
#                 # 处理后缀（如 -None）
#                 if out_chan.startswith(orig_chan):
#                     renamed_list.append(channel_to_new[orig_chan])
#                     found = True
#                     logger(f"[DEBUG] Matched {orig_chan} -> {out_chan} -> {channel_to_new[orig_chan]}")
#                     break
#             if not found:
#                 logger(f"[WARNING] Channel {orig_chan} not found in output")
        
#         # 更新通道名称
#         header['channel_names'] = renamed_list
#         logger(f"[*] Final channel names: {renamed_list}")
#     else:
#         header['channel_names'] = header['channel_names'].original_names
#     # =================================

#     logger("[*] Extracted {} channels: {}".format(psg.shape[1],
#                                                   header['channel_names']))
#     to_h5_file(out_path, psg, **header)



def extract(files, out_dir, channels, renamed_channels, logger, args):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, file_ in enumerate(files):
        if args.use_dir_names:
            name = os.path.split(os.path.split(file_)[0])[-1]
        else:
            name = os.path.splitext(os.path.split(file_)[-1])[0]
        logger("------------------")
        logger("[*] {}/{} Processing {}".format(i + 1, len(files), name))
        out_dir_subject = os.path.join(out_dir, name)
        if not os.path.exists(out_dir_subject):
            os.mkdir(out_dir_subject)
        out_path = os.path.join(out_dir_subject, name + ".h5")
        if os.path.exists(out_path):
            if not args.overwrite:
                logger("-- Skipping (already exists, overwrite=False)")
                continue
            os.remove(out_path)
        _extract(
            file_=file_,
            out_path=out_path,
            channels=channels,
            renamed_channels=renamed_channels,
            logger=logger,
            args=args
        )


def run(args):
    files = glob(args.file_regex)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    logger = Logger(out_dir,
                    active_file='extraction_log',
                    overwrite_existing=args.overwrite,
                    print_calling_method=False)
    logger("Args dump: {}".format(vars(args)))
    logger("Found {} files matching glob statement".format(len(files)))
    if len(files) == 0:
        return
    channels = ChannelMontageTuple(args.channels, relax=True)
    renamed_channels = args.rename_channels
    if renamed_channels and (len(renamed_channels) != len(channels)):
        raise ValueError("--rename_channels argument must have the same number"
                         " of elements as --channels. Got {} and {}.".format(
            len(channels), len(renamed_channels)
        ))

    logger("Extracting channels {}".format(channels.names))
    if renamed_channels:
        logger("Saving channels under names {}".format(renamed_channels))
    logger("Saving .h5 files to '{}'".format(out_dir))
    logger("Re-sampling: {}".format(args.resample))
    logger("-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-")
    extract(
        files=files,
        out_dir=out_dir,
        channels=channels,
        renamed_channels=renamed_channels,
        logger=logger,
        args=args
    )


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    run(args)


if __name__ == "__main__":
    entry_func()
