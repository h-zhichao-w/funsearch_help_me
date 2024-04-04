"""
规划卫星轨道和成像任务
"""
import numpy as np
from tqdm import tqdm


def mission_plan(cell_strip, open_num_max, open_orbit_time_max,survey_orbit_num_max, survey_orbit_time_max, open_time_max,open_time_min, open_inter, survey_open_num_max, survey_open_time_max,survey_time_max, survey_time_min, survey_inter_same, survey_inter_diff):
    """
    Args:
        cell_strip: 2D array, 其大小为 orbitNum 行 waveNum 列，其中 orbitNum 表示轨道数，waveNum 表示波位数，存储有每一轨每个波位的条带。
        open_num_max: int, 最大开机次数。
        open_orbit_time_max: int, 每轨最大开机累计时间。
        survey_orbit_num_max: int, 每轨最大成像次数。
        survey_orbit_time_max: int, 每轨最大成像累计时间。
        open_time_max: int, 开机模式的最大持续时间。
        open_time_min: int, 开机模式的最小持续时间。
        open_inter: int, 开机间隔时间。
        survey_open_num_max: int, 单次开机最大成像次数。
        survey_open_time_max: int, 单次开机最大累计成像时间。
        survey_time_max: int, 成像最大时长。
        survey_time_min: int, 成像最小时长。
        survey_inter_same: int, 同波段成像时间间隔。
        survey_inter_diff: int, 不同波段成像时间间隔。
    """
    orbit_num, wave_num = cell_strip.shape
    # 提取升降轨标识
    sj = np.zeros_like(cell_strip, dtype=int)
    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            if type(cell_strip[orbit, wave]) != int:
                sj[orbit, wave] = cell_strip[orbit, wave][0][2]

    for orbit in tqdm(range(orbit_num)):



