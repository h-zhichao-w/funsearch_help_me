"""
规划卫星轨道和成像任务
"""
import numpy as np
from tqdm import tqdm
from PlanOrbit import plan_orbit


def mission_plan(cell_strip, **kwargs):
    """
    Args:
        cell_strip: 2D array, 其大小为 orbitNum 行 waveNum 列，其中 orbitNum 表示轨道数，waveNum 表示波位数，存储有每一轨每个波位的条带。
        kwargs: 开机和观测约束条件
    """

    orbit_num, wave_num = cell_strip.shape

    orbit_plan = [[] for _ in range(orbit_num)]
    survey_plan = [[] for _ in range(orbit_num)]

    # 提取升降轨标识
    sj = np.zeros((orbit_num, ), dtype=int)
    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            if len(cell_strip[orbit, wave]):
                sj[orbit] = cell_strip[orbit, wave][0][2]

    for orbit in tqdm(range(orbit_num)):
        strips = cell_strip[orbit]
        # if not np.all(strips):
        #     continue
        row_num = max(len(wave) for wave in strips if type(wave) is not int)
        strip_of_orbit = [[] for _ in range(row_num)]
        for row in range(row_num):
            for wave in range(wave_num):
                try:
                    strip_of_orbit[row].append(list(strips[wave][row]))
                except IndexError:
                    strip_of_orbit[row].append([np.inf] * 3)
        strip_of_orbit = np.array(strip_of_orbit)
        for i in range(strip_of_orbit.shape[0]):
            for j in range(strip_of_orbit.shape[1]):
                if strip_of_orbit[i, j, 0] != np.inf:
                    strip_of_orbit[i, j, 2] = j + 1

        if strip_of_orbit.dtype == np.dtype('int32'):
            strip_of_orbit = strip_of_orbit.astype('float64')

        survey_total, orbit_total = plan_orbit(strip_of_orbit, **kwargs)

        survey_plan[orbit] = [[] for _ in range(len(orbit_total))]
        # 遍历重访周期
        for k in range(len(orbit_total)):
            orbit_strip = np.array(orbit_total[k]['strip'], dtype=int)
            orbit_plan[orbit].append([orbit_strip[:, :3], sj[orbit]])
            no = orbit_strip[:, 0]


            # temp = []
            for p in no:
                strip_p = np.array(survey_total[p]['strip'], dtype=int)
                strip_p = np.hstack((strip_p, np.ones((strip_p.shape[0], 1)) * sj[orbit]))
                survey_plan[orbit][k].append(strip_p)

            survey_plan[orbit][k] = np.vstack((survey_plan[orbit][k][:]))
            # survey_plan[orbit].append(temp)

    survey_plan = np.array(survey_plan, dtype=object)
    orbit_plan = np.array(orbit_plan, dtype=object)

    return orbit_plan, survey_plan, sj
