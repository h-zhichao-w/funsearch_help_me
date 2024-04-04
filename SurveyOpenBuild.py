import numpy as np

def survey_open_build(strips_of_orbit, **kwargs):
    """
    Args:
        strips_of_orbit: 2D array, 其大小为 1 行 waveNum 列，其中waveNum 表示波位数，存储有每一轨每个波位的条带。
        **kwargs: dict, 其中包括了规划轨道和成像任务的约束条件。

    """

    # 将开机信息保存为一个字典
    survey = {'stips': [], 'number': 0, 'inter': [], 'time': 0, 'open': 0}
    orbit_num, wave_num = strips_of_orbit.shape
    sign_inf = np.full((orbit_num, wave_num), float('inf'))

    while not (strips_of_orbit == sign_inf):
        if survey['number'] == 0:
            # 找到最早的一个任务
            index = np.where(strips_of_orbit[:, 0] == np.min(strips_of_orbit[:, 0]))[0][0]  # index是一个元组，里面是一个np.ndarry，提取两次才能得到数
            plan_end_pre = strips_of_orbit[index, 1]
            plan_st_pre = strips_of_orbit[index, 0]
        else:
            index_same = np.where(strips_of_orbit[:, 2] == plan_wave)
            index_diff = np.where(strips_of_orbit[:, 2] != plan_wave)
            survey_gap = np.zeros_like(strips_of_orbit)
            survey_gap[index_same, 0] = strips_of_orbit[index_same, 0] - plan_end - 1 - kwargs['survey_inter_same']
            survey_gap[index_diff, 0] = strips_of_orbit[index_diff, 0] - plan_end - 1 - kwargs['survey_inter_diff']
            temp = np.where(survey_gap[:, 0] >= 0)
            if len(temp) == 0:
                break
            else:
                index = np.where(strips_of_orbit[:, 0] == min(strips_of_orbit[temp, 0]))[0][0]

        plan_st = strips_of_orbit[index, 0]
        plan_end = strips_of_orbit[index, 1]
        plan_wave = strips_of_orbit[index, 2]

        if survey['time'] < kwargs['survey_open_time_max'] and survey['open'] < kwargs['open_time_max']:
            # 成像次数约束
            if survey['number'] < 4:
                # 最小成像时长约束
                if plan_end - plan_st + 1 < kwargs['survey_time_min']:
                    if survey['number'] == 0:   # 补满
                        plan_end = plan_st + kwargs['survey_time_min'] - 1
                    else:
                        if plan_wave == plan_wave_pre:
                            survey_inter = kwargs['survey_inter_same']
                        else:
                            survey_inter = kwargs['survey_inter_diff']
                        if plan_end_pre + survey_inter + 1 == plan_st:
                            plan_end = plan_st + kwargs['survey_time_min'] - 1
                        elif plan_end_pre + survey_inter + 1 < plan_st:

