"""
处理栅格数据gridDataLX中的线段，将持续时间小于等于max(surveyTimeMin,openTimeMin)的线段，合并转为cellStripLX，其中每个cell都由[持续线段的起始采样点，持续线段的结束采样点，升降轨]组成
"""

import numpy as np
from tqdm import tqdm

def strip2stripLX(grid_data, survey_time_min, open_time_min):
    """
    Args:
        grid_data: 2D array, 其大小为 orbitNum 行 waveNum 列，其中 orbitNum 表示轨道数，waveNum 表示波位数。
        survey_time_min: int, 调查模式的最小持续时间。
        open_time_min: int, 开机模式的最小持续时间。
    """
    orbit_num, wave_num = grid_data.shape
    cell_strip_LX = np.zeros((orbit_num, wave_num), dtype=object)

    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            strips = grid_data[orbit, wave]
            if strips == 0:
                continue
            # 若只有一个条带，则直接加入
            if len(strips) == 1:
                cell_strip_LX[orbit, wave] = strips
            # 若有多个条带，则检查合并
            else:
                pst = 0
                pend = 1
                lx = []
                for i in range(len(strips) - 1):
                    if strips[pend][0] - strips[pst][0] < max(survey_time_min, open_time_min):
                        pend += 1
                    else:
                        lx.append([strips[pst][0], strips[pend - 1][1], strips[pst][2]])
                        pst = pend
                        pend += 1
                    if pend == len(strips):
                        lx.append([strips[pst][0], strips[pend - 1][1], strips[pst][2]])
                        break

                cell_strip_LX[orbit, wave] = lx

    # 将cell_strip_LX中每个元素转换为numpy数组
    strips = np.zeros_like(cell_strip_LX, dtype=object)
    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            if cell_strip_LX[orbit, wave]:
                strips[orbit, wave] = np.array(cell_strip_LX[orbit, wave])

    return strips
