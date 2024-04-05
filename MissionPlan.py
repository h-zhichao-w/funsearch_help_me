"""
规划卫星轨道和成像任务
"""
import numpy as np
from tqdm import tqdm


def mission_plan(cell_strip, **kwargs):
    """
    Args:
        cell_strip: 2D array, 其大小为 orbitNum 行 waveNum 列，其中 orbitNum 表示轨道数，waveNum 表示波位数，存储有每一轨每个波位的条带。
        kwargs: 开机和观测约束条件
    """
    orbit_num, wave_num = cell_strip.shape
    # 提取升降轨标识
    sj = np.zeros_like(cell_strip, dtype=int)
    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            if type(cell_strip[orbit, wave]) != int:
                sj[orbit, wave] = cell_strip[orbit, wave][0][2]

    for orbit in tqdm(range(orbit_num)):
        pass

    return 0, 0



