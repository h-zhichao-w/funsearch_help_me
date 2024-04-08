from typing import List, Any

import numpy as np


def open_orbit_build(survey_total_array, **kwargs):

    open_build = {'strip': [], 'open_num': 0, 'survey_num': 0, 'inter': [], 'open_time': 0, 'survey_time': 0}
    open_end_pre = 0

    while not np.isinf(survey_total_array).all():
        if open_build['open_num'] == 0:
            index_c = [i for i in range(survey_total_array.shape[0]) if survey_total_array[i, 1] == min(survey_total_array[:, 1])][0]
            open_end_pre = survey_total_array[index_c, 2]

        else:
            if open_build['open_num'] < kwargs['open_num_max']:
                index_c = [i for i in range(survey_total_array.shape[0]) if survey_total_array[i, 1] >= open_end_pre + kwargs['open_inter'] + 1 and survey_total_array[i, 3] <= kwargs['survey_orbit_num_max'] - open_build['survey_num'] and survey_total_array[i, 4] <= kwargs['survey_orbit_time_max'] - open_build['survey_time']]
            else:
                break

        # 这里index_c可能是数也可能是列表，只有index_c是空列表的时候才推出循环
        if np.array(index_c).size:
            if type(index_c) is int:
                list_index_c = [index_c]
            else:
                list_index_c = index_c
            array_temp: list[list[int | Any] | list[Any]] = []
            for i in range(len(list_index_c)):
                open_st = survey_total_array[list_index_c[i], 1]
                open_end = survey_total_array[list_index_c[i], 2]
                open_time = survey_total_array[list_index_c[i], 5]
                if open_time < kwargs['open_time_min']:
                    if open_build['open_num'] == 0:
                        open_end = open_st + kwargs['open_time_min'] - 1
                    else:
                        if open_end_pre + kwargs['open_inter'] + 1 == open_st:
                            open_end = open_st + kwargs['open_time_min'] - 1
                        elif open_end_pre + kwargs['open_inter'] + 1 < open_st:
                            if open_end - kwargs['open_time_min'] + 1 <= open_end_pre + kwargs['open_inter']:
                                open_st = open_end_pre + kwargs['open_inter'] + 1
                                open_end = open_st + kwargs['open_time_min'] - 1
                            else:
                                open_st = open_end - kwargs['open_time_min'] + 1
                    array_temp.append([survey_total_array[list_index_c[i], 0], open_st, open_end, survey_total_array[list_index_c[i], 3], survey_total_array[list_index_c[i], 4], kwargs['open_time_min']])

                else:
                    array_temp.append(list(survey_total_array[list_index_c[i]]))

            index_m = [i for i in range(len(array_temp)) if array_temp[i][5] <= kwargs['open_orbit_time_max'] - open_build['open_time']]
            if not len(index_m):
                break

            index_f = [i for i in index_m if array_temp[i][1] == min([array_temp[j][1] for j in index_m])][0]
            no = int(array_temp[index_f][0])
            open_build['strip'].append(tuple(array_temp[index_f]))
            open_build['open_num'] += 1
            open_build['survey_num'] += array_temp[index_f][3]
            open_build['inter'].append(array_temp[index_f][1] - open_end_pre)
            open_end_pre = array_temp[index_f][2]
            open_build['open_time'] += array_temp[index_f][5]
            open_build['survey_time'] += array_temp[index_f][4]
            survey_total_array[no] = np.inf
            continue
        else:
            break

    return open_build, survey_total_array
