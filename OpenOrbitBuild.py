from typing import List, Any

import numpy as np


def open_orbit_build(survey_total_array, **kwargs):

    open_build = {'strip': [], 'open_num': 0, 'survey_num': 0, 'inter': [], 'open_time': 0, 'survey_time': 0}
    open_end_pre = 0

    while not np.isinf(survey_total_array).all():
        if open_build['open_num'] == 0:
            index_c = np.where(survey_total_array[:, 1] == np.min(survey_total_array[:, 1]))
            open_end_pre = survey_total_array[index_c[0], 2]

        else:
            if open_build['open_num'] < kwargs['open_num_max']:
                index_c = np.where(survey_total_array[:, 1] >= open_end_pre + kwargs['open_inter'] + 1 and survey_total_array[:, 4] <= kwargs['survey_orbit_num_max'] - open_build['survey_num'] and survey_total_array[:, 5] <= kwargs['survey_orbit_time_max'] - open_build['survey_time'])
            else:
                break

        if len(index_c[0]):
            array_temp: list[list[int | Any] | list[Any]] = []
            for i in range(len(index_c[0])):
                open_st = survey_total_array[index_c[0][i], 1]
                open_end = survey_total_array[index_c[0][i], 2]
                open_time = survey_total_array[index_c[0][i], 5]
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
                    array_temp.append([survey_total_array[index_c[0][i], 0], open_st, open_end, survey_total_array[index_c[0][i], 3], survey_total_array[index_c[0][i], 4], kwargs['open_time_min']])

                else:
                    array_temp.append(list(survey_total_array[index_c[0][i]]))

            index_m = [i for i in range(len(array_temp)) and array_temp[i][5] <= kwargs['open_orbit_time_max'] - open_build['open_time']]
            if not len(index_m):
                break

            index_f = [i for i in index_m and array_temp[i][2] == min([array_temp[j][2] for j in index_m])][0]
            no = array_temp[index_f][0]
            open_build['strip'].append(list(survey_total_array[index_f]))
            open_build['open_num'] += 1
            open_build['survey_num'] += array_temp[index_f][3]
            open_build['inter'].append(array_temp[index_f][1] - open_end_pre)
            open_end_pre = array_temp[index_f][2]
            open_build['open_time'] += array_temp[index_f][5]
            open_build['survey_time'] += array_temp[index_f][4]
            survey_total_array[no] = np.inf
            continue

    return open_build, survey_total_array
