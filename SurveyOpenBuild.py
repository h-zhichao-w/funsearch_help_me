import numpy as np

def survey_open_build(strips_of_orbit, **kwargs):
    """
    Args:
        strips_of_orbit: 2D array, 其大小为 1 行 waveNum 列，其中waveNum 表示波位数，存储有每一轨每个波位的条带。
        **kwargs: dict, 其中包括了规划轨道和成像任务的约束条件
            survey_time_min: int, 调查模式的最小持续时间。



    """

    # 将开机信息保存为一个字典
    survey = {'stips': [], 'number': 0, 'inter': [], 'time': 0, 'open': 0}
    sign_inf = np.zeros_like(strips_of_orbit)
    # 初始化部分变量（没有用，只是为了符合语法，原程序的代码习惯太糟糕了）
    plan_st = 0
    plan_end = 0
    plan_wave = 0
    plan_st_pre = 0
    plan_end_pre = 0
    plan_wave_pre = 0
    plan0 = 0


    while not (strips_of_orbit == sign_inf):
        # 若当前尚未安排任务（该分支进且只会进一次）
        if survey['number'] == 0:
            # 找到最早的一个任务
            index = np.where(strips_of_orbit[:, 0] == np.min(strips_of_orbit[:, 0]))[0][0]  # index是一个元组，里面是一个numpy数组，提取两次才能得到数
            # 由于这是第一个考虑的任务，所以对后续而言，这个一定是“上一个”任务
            plan_end_pre = strips_of_orbit[index, 1]
            plan_st_pre = strips_of_orbit[index, 0]
            plan_wave_pre = strips_of_orbit[index, 2]
            plan0 = plan_st_pre # 开机时刻
        # 第二次进循环开始，一定会进该分支
        else:
            # 与上一个任务相同的波段和不同的波段的条带编号
            index_same = np.where(strips_of_orbit[:, 2] == plan_wave)
            index_diff = np.where(strips_of_orbit[:, 2] != plan_wave)
            # 用survey_gap记录每个任务和上个任务之间会需要的时间间隔
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

        # 开机时间小于最大开机时间且开机次数小于最大开机次数
        if survey['time'] < kwargs['survey_open_time_max'] and survey['open'] < kwargs['open_time_max']:
            # 成像次数约束
            if survey['number'] < 4:
                # 任务持续时间小于最小持续时间
                if plan_end - plan_st + 1 < kwargs['survey_time_min']:
                    if survey['number'] == 0:   # 如果这是第一个任务，向后补满时间
                        plan_end = plan_st + kwargs['survey_time_min'] - 1
                    else:
                        # 先决定间隔时间
                        if plan_wave == plan_wave_pre:
                            survey_inter = kwargs['survey_inter_same']
                        else:
                            survey_inter = kwargs['survey_inter_diff']

                        # 间隔时间正好
                        if plan_end_pre + survey_inter + 1 == plan_st:
                            plan_end = plan_st + kwargs['survey_time_min'] - 1
                        # 间隔时间太多
                        elif plan_end_pre + survey_inter + 1 < plan_st:
                            # 如果结束时间不变的情况下，开始时间太早，那么平移整个条带
                            if plan_end - kwargs['survey_time_min'] + 1 <= plan_end_pre + survey_inter + 1:
                                plan_st = plan_end_pre + survey_inter + 1
                                plan_end = plan_st + kwargs['survey_time_min'] - 1
                            # 如果结束时间不变的情况下，开始时间满足间隔要求，则只改变开始时间
                            else:
                                plan_st = plan_end - kwargs['survey_time_min'] + 1

                    # 当前条带结束满足最大开机时长的约束
                    if plan_end - plan0 + 1 <= kwargs['open_time_max']:
                        # 满足最大成像时长约束
                        if survey['time'] + plan_end - plan_st + 1 <= kwargs['survey_open_time_max']:
                            # 记录条带
                            survey['strip'].append([plan_st, plan_end, plan_wave])
                            survey['number'] += 1
                            survey['inter'].append(plan_st - plan_end_pre - 1)
                            plan_end_pre = plan_end
                            plan_wave_pre = plan_wave
                            survey['time'] += plan_end - plan_st + 1
                            survey['open'] += plan_end - plan0 + 1
                            strips_of_orbit[index, :] = 0
                            print('=' * 20)
                            print('当前条带：', [plan_st, plan_end, plan_wave])
                            print('当前成像次数：', survey['number'])
                            print('='*20)
                            continue
                        else:
                            break
                    else:
                        break

                # 任务持续时间大于等于最小持续时间
                else:
                    # 当前条带结束满足最大开机时长的约束
                    if plan_end - plan0 + 1 <= kwargs['open_time_max']:
                        # 满足累计成像时长约束
                        if survey['time'] + plan_end - plan_st + 1 <= kwargs['survey_open_time_max']:
                            # 满足最大成像时长约束
                            if plan_end - plan_st + 1 <= kwargs['survey_time_max']:
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['open'] += plan_end - plan0 + 1
                                strips_of_orbit[index, :] = 0
                                print('=' * 20)
                                print('当前条带：', [plan_st, plan_end, plan_wave])
                                print('当前成像次数：', survey['number'])
                                print('=' * 20)
                                continue
                            else:
                                plan_end = plan_st + kwargs['survey_time_max'] - 1
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['open'] += plan_end - plan0 + 1
                                strips_of_orbit[index, :] = 0
                                print('=' * 20)
                                print('当前条带：', [plan_st, plan_end, plan_wave])
                                print('当前成像次数：', survey['number'])
                                print('=' * 20)
                                continue
                        # 超出累计成像时长约束
                        else:
                            plan_end = plan_st + kwargs['survey_open_time_max'] - survey['time'] - 1
                            # 满足最大成像时长约束
                            if plan_end - plan_st + 1 <= kwargs['survey_time_max']:
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['open'] += plan_end - plan0 + 1
                                strips_of_orbit[index, :] = 0
                                print('=' * 20)
                                print('当前条带：', [plan_st, plan_end, plan_wave])
                                print('当前成像次数：', survey['number'])
                                print('=' * 20)
                                continue
                            else:
                                plan_end = plan_st + kwargs['survey_time_max'] - 1
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['open'] += plan_end - plan0 + 1
                                strips_of_orbit[index, :] = 0
                                print('=' * 20)
                                print('当前条带：', [plan_st, plan_end, plan_wave])
                                print('当前成像次数：', survey['number'])
                                print('=' * 20)
                                continue




