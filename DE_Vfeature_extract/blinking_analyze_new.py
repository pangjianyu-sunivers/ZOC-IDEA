import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import re


# Goal: compare every frame against a 6-second window. Frames whose normalised opening
# leaves the "regular range" (a scaled mix of local and global mean +- std) are flagged
# as part of a blink.
local_region = 4
freq = 29
# canshu = 3.7  # legacy default
canshu = 1.6


def find_local_threshold(lst, frame, canshu):
    """Compute the local lower threshold for a 4-second window centred on ``frame``."""

    # Edge case: handle frames within the first half-window
    if frame < local_region / 2 * freq:
        start_frame = 0
        end_frame = local_region * freq
    else:
        start_frame = frame - local_region / 2 * freq
        end_frame = frame + local_region / 2 * freq
    start_frame = int(start_frame)
    end_frame = int(end_frame)

    up_threshold = 1
    down_threshold = np.mean(lst[start_frame:end_frame]) - (np.std(lst) * canshu)

    return up_threshold, down_threshold


def video_feature_extract(folder_path):
    """Convert per-frame eye-opening series into per-video high-level blink features."""

    df_out = pd.read_excel('/data/DE_Vfeature_extract/blinking_analyze_start.xlsx')
    df = pd.read_excel(f'{folder_path}/blinking_series.xlsx')
    # print(df.at[0,'blinking'])

    for index, row in df.iterrows():

        # print(type(df.at[0,'blinking']))

        # Strip the brackets and split on commas
        string = row['blinking'].strip("[]")
        # Convert the comma-separated string into a list of floats
        if len(string.split(", ")) > 1:
            lst = list(map(float, string.split(", ")))
        else:
            continue

        # print(lst[0])
        # print(max(lst),min(lst),np.mean(lst))

        max_blink = max(lst)
        min_blink = min(lst)
        std_blink = np.std(lst)
        print(row['video_name'])
        df_out.at[index, 'video_name'] = row['video_name']
        pid_match = re.search(r"pid_([^_]+)", str(row['video_name']))
        if pid_match:
            df_out.at[index, 'pid'] = pid_match.group(1).strip()
        else:
            print('Bad video filename: missing pid')
        osdi_match = re.search(r"OSDI_([0-9]+(?:\.[0-9]+)?)", str(row['video_name']))
        if osdi_match:
            df_out.at[index, 'OSDI'] = osdi_match.group(1).strip()
        else:
            print('Bad video filename: missing OSDI')
        # print(std_blink)

        # if min_blink/max_blink > 0.65:
        # 	close_eye_level = 0.55 * max_blink
        # else:
        # 	close_eye_level = min_blink

        if std_blink > 2.7:
            # close_eye_level = 0
            print('low_quality')
            df_out.at[index, 'video_quality'] = 'low_quality'
            # continue
        # 	open_eye_level = std_blink * max_blink
        # 	blink_scale = open_eye_level - close_eye_level
        # 	canshu = 8
        # else:
        # 	close_eye_level = 0
        # 	open_eye_level = max_blink
        # 	blink_scale = max_blink - close_eye_level
        # 	canshu = 3.7
        # print(open_eye_level,close_eye_level,blink_scale)

        open_eye_level = max_blink
        blink_scale = max_blink
        close_eye_level = 0
        canshu = 1.6

        # Plotting buffers
        up_threshold_list = []
        down_threshold_list = []
        open_eye_rate_list = []
        seconds = []
        blinking_label_list = []
        step = 0


        # Buffers for downstream feature extraction
        half_blink_count = 0
        blink_count = 0
        blinking_time_used_list = []
        close_eye_time_used_list = []
        open_eye_time_used_list = []
        blink_start = False
        reach_bottom = False


        for blink_level in lst:

            # Build the open-rate trace
            open_eye_rate = (blink_level - close_eye_level) / blink_scale
            # print(open_eye_rate)
            open_eye_rate_list.append(open_eye_rate)
            time_seconds = step / freq
            seconds.append(time_seconds)
            step = step + 1

        # print(np.std(open_eye_rate_list))
        step = 0
        for blink in open_eye_rate_list:
            normal_count = abs(blink - np.mean(open_eye_rate_list))
            up_threshold, down_threshold = find_local_threshold(open_eye_rate_list, step, canshu)
            # if normal_count > 0.12 or blink > up_threshold or blink < down_threshold:
            # if normal_count > 0.12 or blink < down_threshold:
            up_threshold_list.append(up_threshold)
            down_threshold_list.append(down_threshold)
            if blink < down_threshold:
                blinking_label_list.append(0.7)
            else:
                blinking_label_list.append(0)
            step = step + 1



        # 	# Legacy blink-parameter accumulation (kept for reference)
        # 	if open_eye_rate >= 0.6:
        # 		if blink_start == True and reach_bottom == False:
        # 			blink_start = False
        # 			half_blink_count += 1
        # 		elif blink_start == True and reach_bottom == True:
        # 			print('3')
        # 			blink_start = False
        # 			reach_bottom = False
        # 			blink_count += 1
        # 			blinking_time_used = time_seconds - blink_start_time
        # 			blinking_time_used_list.append(blinking_time_used)
        # 			open_eye_time_used = time_seconds - open_eye_start
        # 			open_eye_time_used_list.append(open_eye_time_used)

        # 	if open_eye_rate < 0.6 and open_eye_rate > 0.2:
        # 		if blink_start == False:
        # 			print('1')
        # 			blink_start = True
        # 			blink_start_time = time_seconds

        # 	if open_eye_rate <= 0.2:
        # 		if blink_start == True:
        # 			print('2')
        # 			reach_bottom = True
        # 			close_eye_time_used = time_seconds - blink_start_time
        # 			close_eye_time_used_list.append(close_eye_time_used)
        # 			open_eye_start = time_seconds
        # 		if blink_start == False:
        # 			print('12')
        # 			blink_start = True
        # 			reach_bottom = True
        # 			blink_start_time = time_seconds

            # x-axis counter
            # step = step + 1

        # Legacy single-pass feature dump (kept commented for reference)
        # df.at[index,'max_palpebral_fissure_height'] = max_blink
        # df.at[index,'min_palpebral_fissure_height'] = min_blink
        # df.at[index,'complete_blink_count'] = blink_count
        # df.at[index,'incomplete_blink_count'] = half_blink_count
        # if blink_count > 1:
        # 	df.at[index,'blink_interval_cv'] = np.std(blinking_time_used_list)/np.mean(blinking_time_used_list)
        # 	df.at[index,'max_blink_interval'] = max(blinking_time_used_list)
        # df.at[index,'mean_raw_blink_amp'] = np.mean(lst)
        # df.at[index,'max_palpebral_fissure_height_raw'] = max_blink
        # if blink_count > 1:
        # 	df.at[index,'mean_blink_duration'] = np.mean(blinking_time_used_list)
        # 	df.at[index,'min_blink_duration'] = min(blinking_time_used_list)
        # 	df.at[index,'min_eye_close_duration'] = min(close_eye_time_used_list)
        # 	df.at[index,'mean_eye_close_duration'] = np.mean(close_eye_time_used_list)
        # 	df.at[index,'min_eye_open_duration'] = min(open_eye_time_used_list)
        # 	df.at[index,'mean_eye_open_duration'] = np.mean(open_eye_time_used_list)


        # plt.plot(seconds, up_threshold_list)
        plt.plot(seconds, down_threshold_list)
        plt.plot(seconds, open_eye_rate_list)
        plt.plot(seconds, blinking_label_list)
        plt.ylim(0, 1)
        # plt.show()
        plt.savefig(f"{folder_path}/{row['video_name']}.png")
        plt.close()

        blink_prev = blinking_label_list[0]
        blink_seconds = ''
        for i, blink_frame in enumerate(blinking_label_list):
            if blink_prev != 0 and blink_frame == 0:
                blink_seconds += f'{i / freq}-'
            elif blink_prev == 0 and blink_frame != 0:
                blink_seconds += f'{i / freq},'
            blink_prev = blink_frame



        ################# Merge consecutive blinks separated by < 0.2 s into a single blink #################
        # Example raw output:
        # 0.06896551724137931 , 0.13793103448275862 - 0.1724137931034483 , 0.20689655172413793 - 8.482758620689655 , 8.655172413793103 - ...
        # 3.0,3.0344827586206895-3.0689655172413794,3.206896551724138-
        blink_seconds_real = ''
        blink_merging = blink_seconds.split(',')
        if len(blink_merging) >= 2:
            for blink_merging_part in blink_merging:
                blink_merging_part_sep = blink_merging_part.split('-')
                if len(blink_merging_part_sep) == 2:
                    # print(blink_merging_part_sep)
                    if '' in blink_merging_part_sep:
                        # print('end')
                        blink_seconds_real += f'{blink_merging_part_sep[0]}'
                        continue
                    sep_time = float(blink_merging_part_sep[1]) - float(blink_merging_part_sep[0])
                    if sep_time < 0.2:
                        blink_seconds_real += ''
                    else:
                        blink_seconds_real += f'{blink_merging_part_sep[0]}-{blink_merging_part_sep[1]},'
                else:
                    blink_seconds_real += f'{blink_merging_part_sep[0]},'
        else:
            blink_seconds_real = blink_seconds

        # print(blink_seconds_real)




        ################# Blink annotation column #################
        df_out.at[index, 'blink_annotation'] = blink_seconds_real

        '''
        blink_interval_cv: std(complete_blink_durations) / mean(complete_blink_durations) — under discussion.

        min_eye_close_duration: time between the first frame the orange line drops below
            the blue threshold and the lowest blink point — minimum across all blinks
            in the video.
        min_eye_open_duration: time between the lowest blink point and the first frame
            the orange line rises above the blue threshold — minimum across all blinks
            in the video.
        '''
        # max/min palpebral fissure height: max/min of the eyelid distance trace
        df_out.at[index, 'max_palpebral_fissure_height'] = max_blink
        df_out.at[index, 'min_palpebral_fissure_height'] = min_blink

        # All inter-blink durations
        sep_time_list = []

        # Complete vs. incomplete blink counts: a blink (orange < blue) lasting >0.2 s
        # counts as complete, otherwise as incomplete.
        complete_blink = 0
        complete_blink_timestep = ''
        incomplete_blink = 0
        incomplete_blink_timestep = ''
        blink_divide = blink_seconds_real.split('-')
        if len(blink_divide) >= 2:
            for blinking in blink_divide:
                blinking_timestep = blinking.split(',')
                if len(blinking_timestep) == 2:
                    # print(blinking_timestep)
                    if '' in blinking_timestep:
                        # print('end')
                        continue
                    sep_time = float(blinking_timestep[1]) - float(blinking_timestep[0])
                    sep_time_list.append(sep_time)
                    if sep_time > 0.2:
                        complete_blink += 1
                        complete_blink_timestep += f'{blinking_timestep[0]},{blinking_timestep[1]}-'
                    else:
                        incomplete_blink += 1
                        incomplete_blink_timestep += f'{blinking_timestep[0]},{blinking_timestep[1]}-'

        # print(complete_blink)
        # print(incomplete_blink)
        df_out.at[index, 'complete_blink_count'] = complete_blink
        df_out.at[index, 'complete_blink_timestamps'] = complete_blink_timestep
        df_out.at[index, 'incomplete_blink_count'] = incomplete_blink
        df_out.at[index, 'incomplete_blink_timestamps'] = incomplete_blink_timestep
        df_out.at[index, 'total_blink_count'] = complete_blink

        # max_blink_interval: largest gap between consecutive blinks across the whole video
        max_blink_interval = 0
        blink_seperation = blink_seconds_real.split(',')
        if len(blink_seperation) >= 2:
            for blinking_seperate in blink_seperation:
                blinking_sep_time = blinking_seperate.split('-')
                if len(blinking_sep_time) == 2:
                    # print(blinking_sep_time)
                    if '' in blinking_sep_time:
                        # print('end')
                        continue
                    blink_interval = float(blinking_sep_time[1]) - float(blinking_sep_time[0])
                    if blink_interval > max_blink_interval:
                        max_blink_interval = blink_interval
            df_out.at[index, 'max_blink_interval'] = max_blink_interval

        # mean_blink_duration: mean of the per-blink durations (orange < blue duration)
        if len(sep_time_list) > 0:
            df_out.at[index, 'mean_blink_duration'] = np.mean(sep_time_list)

        # min_blink_duration: minimum of the per-blink durations
        if len(sep_time_list) > 0:
            df_out.at[index, 'min_blink_duration'] = min(sep_time_list)

        # min_eye_close_duration / min_eye_open_duration:
        #   For each blink, the closing duration is from the first sub-threshold frame
        #   to the lowest blink point; the opening duration is from the lowest blink
        #   point to the first super-threshold frame. We track the per-video min/max.
        close_eye_time_min = 3
        open_eye_time_min = 3
        close_eye_time_max = 0
        open_eye_time_max = 0
        blink_divide = blink_seconds_real.split('-')
        if len(blink_divide) >= 2:
            for blinking in blink_divide:
                blinking_timestep = blinking.split(',')
                if len(blinking_timestep) == 2:
                    # print(blinking_timestep)
                    if '' in blinking_timestep:
                        # print('end')
                        continue
                    starting = round(float(blinking_timestep[0]) * freq)
                    ending = round(float(blinking_timestep[1]) * freq)
                    this_blink = lst[starting:ending]
                    # print(starting,ending,this_blink)
                    low_point_index = this_blink.index(min(this_blink))
                    close_eye_time = low_point_index / freq
                    open_eye_time = (len(this_blink) - low_point_index) / freq
                    if close_eye_time < close_eye_time_min:
                        close_eye_time_min = close_eye_time
                    if open_eye_time < open_eye_time_min:
                        open_eye_time_min = open_eye_time
                    if close_eye_time > close_eye_time_max:
                        close_eye_time_max = close_eye_time
                    if open_eye_time > open_eye_time_max:
                        open_eye_time_max = open_eye_time
            df_out.at[index, 'min_eye_close_duration'] = close_eye_time_min
            df_out.at[index, 'min_eye_open_duration'] = open_eye_time_min
            df_out.at[index, 'max_eye_close_duration'] = close_eye_time_max
            df_out.at[index, 'max_eye_open_duration'] = open_eye_time_max




    if 'pid' in df_out.columns:
        df_out['pid'] = df_out['pid'].apply(lambda value: '' if pd.isna(value) else str(value).strip())
    df_out.to_excel(f'{folder_path}/blinking_results.xlsx', index=False)
