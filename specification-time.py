"""
Specification for task allocation problem. The goal is to allocate tasks to the Strict Return to Orbit (SRO) satellites in a way that minimises the total cycles and time taken to complete all tasks.
"""
# Repositories for solving the task allocation problem.
import scipy.io as sio
import numpy as np
from tqdm import tqdm


def main(dataset: dict) -> tuple[int, int, np.ndarray]:
    """
    Main function for task allocation for the Strict Return to Orbit (SRO) satellites.
    All the parameters related with time are given in the unit of seconds.

    Args:
        dataset: dict, all the data needed for task allocation, including
            dataset['task_name']: str, the name of the task.
            dataset['grid_path']: str, the path of the grid file.
            dataset['boot_num_max']: int, the maximum times of boot-ups in each orbit.
            dataset['boot_orbit_time_max']: int, the maximum cumulative time of boot-ups on in each orbit.
            dataset['survey_orbit_num_max']: int, the maximum number of surveys in each orbit.
            dataset['survey_orbit_time_max']: int, the maximum cumulative time of survey in each orbit.
            dataset['boot_time_max']: int, the maximum lasting time of each boot-up.
            dataset['boot_time_min']: int, the minimum lasting time of each boot-up.
            dataset['boot_inter']: int, the minimum interval between two boot-ups.
            dataset['survey_boot_num_max']: int, the maximum number of surveys in each boot-up.
            dataset['survey_boot_time_max']: int, the maximum cumulative time of surveys in each boot-up.
            dataset['survey_time_max']: int, the maximum lasting time of each survey.
            dataset['survey_time_min']: int, the minimum lasting time of each survey.
            dataset['survey_inter_same']: int, the minimum interval between two surveys that are in the same wave.
            dataset['survey_inter_diff']: int, the minimum interval between two surveys that are in different waves.

    Returns:
        tuple[int, int], the total cycles and time taken to complete all tasks.
            cycle: int, the total cycles taken. A cycle is defined as the satellite passing through all the orbits in one revisiting period, and starting to revisit the same orbits agian.
            time: int, the total time taken.
    """

    # Load the grid data from mat file
    grid_data = sio.loadmat(dataset['grid_path'] + dataset['task_name'] + '.mat')
    grid_data = np.array(grid_data['gridData'])

    # Connect the grids and form the strips
    grid_data_cont = point2strip(grid_data)
    strip_cont = strip2strip_cont(grid_data_cont, dataset['survey_time_min'], dataset['boot_time_min'])

    # get the total number of orbits and wave
    orbit_num, wave_num = strip_cont.shape

    # Initialize the orbit plan and survey plan
    orbit_plan = [[] for _ in range(orbit_num)]
    survey_plan = [[] for _ in range(orbit_num)]

    # Extract the ascending and descending orbit identification
    ad = np.zeros((orbit_num,), dtype=int)
    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            if len(strip_cont[orbit, wave]):
                ad[orbit] = strip_cont[orbit, wave][0][2]

    # Planning orbit by orbit
    for orbit in tqdm(range(orbit_num)):
        strips = strip_cont[orbit]
        # In each wave, there can be more than one strip, we need to extract all the strips in each wave.
        row_num = max(max(len(wave) for wave in strips), 1)
        # Put all the strips in one wave together.
        # The strip_of_orbit is a 3D array, with the size of (row_num, wave_num, 3), and is the basis of planning.
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
                # If the strip is not empty, set the wave number.
                if strip_of_orbit[i, j, 0] != np.inf:
                    strip_of_orbit[i, j, 2] = j + 1

        # Make sure the digital type of the array align with the np.inf
        if strip_of_orbit.dtype == np.dtype('int32'):
            strip_of_orbit = strip_of_orbit.astype('float64')

        # Find the all possible surveys in this orbit
        survey_total = []
        while True:
            survey, strips = survey_boot_build(strip_of_orbit, dataset)
            if len(survey['strip']):
                survey['start'] = survey['strip'][0][0]
                survey['end'] = survey['strip'][survey['number'] - 1][1]
                survey_total.append(survey)
            else:
                break

        survey_total_array = np.zeros((len(survey_total), 6))
        for i in range(len(survey_total)):
            survey_total_array[i, 0] = i
            survey_total_array[i, 1] = survey_total[i]['start']
            survey_total_array[i, 2] = survey_total[i]['end']
            survey_total_array[i, 3] = survey_total[i]['number']
            survey_total_array[i, 4] = survey_total[i]['time']
            survey_total_array[i, 5] = survey_total_array[i, 2] - survey_total_array[i, 1]

        orbit_total = []
        while True:
            boot_build, survey_total_array = boot_orbit_build(survey_total_array, dataset)
            if boot_build['strip']:
                boot_build['start'] = boot_build['strip'][0][1]
                boot_build['end'] = boot_build['strip'][boot_build['boot_num'] - 1][2]
                orbit_total.append(boot_build)
            else:
                break

        survey_plan[orbit] = [[] for _ in range(len(orbit_total))]
        # 遍历重访周期
        for k in range(len(orbit_total)):
            orbit_strip = np.array(orbit_total[k]['strip'], dtype=int)
            orbit_plan[orbit].append([orbit_strip[:, :3], ad[orbit]])
            no = orbit_strip[:, 0]

            for p in no:
                strip_p = np.array(survey_total[p]['strip'], dtype=int)
                strip_p = np.hstack((strip_p, np.ones((strip_p.shape[0], 1)) * ad[orbit]))
                survey_plan[orbit][k].append(strip_p)

            survey_plan[orbit][k] = np.vstack((survey_plan[orbit][k][:]))

    survey_plan = np.array(survey_plan, dtype=object)
    orbit_plan = np.array(orbit_plan, dtype=object)

    no_a = np.where(ad[:] == 1)
    orbit_num = survey_plan.size
    col_num = max(len(orbit_plan[i]) for i in range(orbit_num))
    orbit_plan_array = np.empty((orbit_num, col_num), dtype=object)
    orbit_plan_array.fill([])
    for i in range(orbit_num):
        for j in range(len(orbit_plan[i])):
            strips = orbit_plan[i][j][0]
            strips = np.hstack((strips, np.ones((strips.shape[0], 1)) * orbit_plan[i][j][1]))
            strips = strips.astype(int)
            orbit_plan_array[i][j] = strips

    cycle_num = max(len(survey_plan[i]) for i in range(orbit_num))
    survey_plan_array = np.empty((orbit_num, cycle_num), dtype=object)
    survey_plan_array.fill([])
    for i in range(orbit_num):
        for j in range(len(survey_plan[i])):
            strips = survey_plan[i][j]
            strips = strips.astype(int)
            # sort the strips by the start time in ascending order
            strips = strips[np.argsort(strips[:, 0])]
            survey_plan_array[i][j] = strips

    time_total = np.zeros(cycle_num)
    for k in range(cycle_num):
        for i in range(len(no_a[0])):
            a = no_a[0][i]
            if type(survey_plan_array[a, k]) is not list:
                time_total[k] += sum(survey_plan_array[a, k][:, 1] - survey_plan_array[a, k][:, 0] + 1)
            else:
                continue

    print(time_total)
    print(sum(time_total))

    return cycle_num, sum(time_total), orbit_plan_array


def point2strip(grid_data: np.ndarray) -> np.ndarray:
    """
    Function to connect the grids and form the strips.

    Args:
        grid_data: 2D array, the size is (number of orbits, number of waves).

    Returns:
        grid_data_cont: 2D array, the size is same as grid_data which is (number of orbits, number of waves), containing the connected strips. "cont" stands for "continuous".
    """

    orbit_num, wave_num = grid_data.shape
    grid_data_cont = np.empty((orbit_num, wave_num), dtype=object)
    grid_data_cont.fill([])

    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            if not grid_data[orbit, wave].size:
                continue
            grid_cell = grid_data[orbit, wave]
            pst = 0
            pend = 1
            LX = []
            while True:
                if pend == grid_cell.shape[0]:
                    LX.append([grid_cell[pst, 0], grid_cell[pend - 1, 0], grid_cell[pst, 2]])
                    break
                if grid_cell[pend, 0] - grid_cell[pend - 1, 0] == 1:
                    pend += 1
                    continue
                if grid_cell[pend, 0] - grid_cell[pend - 1, 0] > 1:
                    LX.append([grid_cell[pst, 0], grid_cell[pend - 1, 0], grid_cell[pst, 2]])
                    pst = pend
                    pend += 1
                    continue
            grid_data_cont[orbit, wave] = LX

    return grid_data_cont


def strip2strip_cont(grid_data_cont: np.ndarray, survey_time_min: int, boot_time_min: int) -> np.ndarray:
    """
    Function to process the strips in grid_data_cont and merge the strips whose lasting time is less than or equal to the maximum of survey_time_min and boot_time_min.

    Args:
        grid_data_cont: 2D array, the size is (number of orbits, number of waves), containing the connected strips.
        survey_time_min: int, the minimum lasting time of each survey.
        boot_time_min: int, the minimum lasting time of each boot-up.

    Returns:
        strips: 2D array, the size is (number of orbits, number of waves), containing the merged strips.

    """

    orbit_num, wave_num = grid_data_cont.shape
    cell_strip_cont = np.empty((orbit_num, wave_num), dtype=object)
    cell_strip_cont.fill([])

    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            strips = grid_data_cont[orbit, wave]
            if not len(strips):
                continue
            # If there is only one strip, no need for merging
            if len(strips) == 1:
                cell_strip_cont[orbit, wave] = strips
            # If there are more than one strip, check for merging
            else:
                pst = 0
                pend = 1
                lx = []
                for i in range(len(strips) - 1):
                    if strips[pend][0] - strips[pst][0] < max(survey_time_min, boot_time_min):
                        pend += 1
                    else:
                        lx.append([strips[pst][0], strips[pend - 1][1], strips[pst][2]])
                        pst = pend
                        pend += 1
                    if pend == len(strips):
                        lx.append([strips[pst][0], strips[pend - 1][1], strips[pst][2]])
                        break

                cell_strip_cont[orbit, wave] = lx

    # Convert each element in cell_strip_cont to numpy array
    strips = np.empty((orbit_num, wave_num), dtype=object)
    strips.fill([])
    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            if cell_strip_cont[orbit, wave]:
                strips[orbit, wave] = np.array(cell_strip_cont[orbit, wave])

    return strips


def survey_boot_build(strips_of_orbit: np.ndarray, dataset: dict) -> tuple[dict, np.ndarray]:
    """
    Function that plans the surveys.

    Args:
        strips_of_orbit: 3D array, the size is (number of rows, number of columns, 3), containing the strips in the given orbit.
        dataset: dict, all the data needed for task allocation, in this function, mainly use the constraints for survey.
            dataset['boot_num_max']: int, the maximum times of boot-ups in each orbit.
            dataset['boot_orbit_time_max']: int, the maximum cumulative time of boot-ups on in each orbit.
            dataset['survey_orbit_num_max']: int, the maximum number of surveys in each orbit.
            dataset['survey_orbit_time_max']: int, the maximum cumulative time of survey in each orbit.
            dataset['boot_time_max']: int, the maximum lasting time of each boot-up.
            dataset['boot_time_min']: int, the minimum lasting time of each boot-up.
            dataset['boot_inter']: int, the minimum interval between two boot-ups.
            dataset['survey_boot_num_max']: int, the maximum number of surveys in each boot-up.
            dataset['survey_boot_time_max']: int, the maximum cumulative time of surveys in each boot-up.
            dataset['survey_time_max']: int, the maximum lasting time of each survey.
            dataset['survey_time_min']: int, the minimum lasting time of each survey.
            dataset['survey_inter_same']: int, the minimum interval between two surveys that are in the same wave.
            dataset['survey_inter_diff']: int, the minimum interval between two surveys that are in different waves.
    """

    # Initialize the dict for recording the survey
    survey = {
        'strip': [],    # The strips in the survey
        'number': 0,    # The number of strips in the survey
        'inter': [],    # The intervals between two strips
        'time': 0,      # The total time taken for the survey
        'boot': 0       # The total time of boot-ups
    }

    # Initialization
    plan_st = 0
    plan_end = 0
    plan_wave = 0
    plan_end_pre = 0
    plan_wave_pre = 0
    plan0 = 0
    row = 0
    col = 0

    while not np.isinf(strips_of_orbit).all():
        # if not strip has been arranged
        if survey['number'] == 0:
            # find the earliest strip
            index = np.where(strips_of_orbit[:, :, 0] == np.min(strips_of_orbit[:, :, 0]))
            row = index[0][0]
            col = index[1][0]
            plan_end_pre = strips_of_orbit[row, col, 1]
            # record the start time of the first strip, also the time of boot-up
            plan0 = strips_of_orbit[row, col, 0]

        else:
            # the index of strips that has the same wave number with the previous strip
            index_same = np.where(strips_of_orbit[:, :, 2] == plan_wave)
            # the index of strips that has different wave number with the previous strip
            inf_mask = np.isinf(strips_of_orbit[:, :, 2])
            not_plan_wave_mask = strips_of_orbit[:, :, 2] != plan_wave
            combined_mask = ~inf_mask & not_plan_wave_mask
            index_diff = np.where(combined_mask)
            # survey_gap is the gap between the end of the previous strip and the start of the current strip
            survey_gap = np.ones_like(strips_of_orbit) * -1
            survey_gap[index_same[0], index_same[1], 0] = strips_of_orbit[index_same[0], index_same[1], 0] - plan_end - 1 - dataset['survey_inter_same']
            survey_gap[index_diff[0], index_diff[1], 0] = strips_of_orbit[index_diff[0], index_diff[1], 0] - plan_end - 1 - dataset['survey_inter_diff']
            # find the strip with the smallest gap, if there is any gap satisfies the constraints
            temp = np.where(survey_gap[:, :,  0] >= 0)
            if len(temp[0]) == 0:
                break
            else:
                index = [i for i in range(len(temp[0])) if strips_of_orbit[temp[0][i], temp[1][i], 0] == min(strips_of_orbit[temp[0], temp[1], 0])]
                row = temp[0][index[0]]
                col = temp[1][index[0]]

        plan_st = strips_of_orbit[row, col, 0]
        plan_end = strips_of_orbit[row, col, 1]
        plan_wave = strips_of_orbit[row, col, 2]

        # if the lasting time of the survey is less than the maximum lasting time of the survey in one boot
        # and the lasting time of the boot-up is less than the maximum lasting time of boot-ups
        if survey['time'] < dataset['survey_boot_time_max'] and survey['boot'] < dataset['boot_time_max']:

            if survey['number'] < dataset['survey_boot_num_max']:
                # the lasting time of the survey is less than the minimum lasting time of the survey
                if plan_end - plan_st + 1 < dataset['survey_time_min']:
                    # if this is the first strip, extend the survey to the minimum lasting time of the survey backward
                    if survey['number'] == 0:
                        plan_end = plan_st + dataset['survey_time_min'] - 1
                    # else, extend the survey based on the interval
                    else:
                        # find the interval
                        if plan_wave == plan_wave_pre:
                            survey_inter = dataset['survey_inter_same']
                        else:
                            survey_inter = dataset['survey_inter_diff']

                        # if the interval is just met
                        if plan_end_pre + survey_inter + 1 == plan_st:
                            plan_end = plan_st + dataset['survey_time_min'] - 1
                        # if the interval is larger than the requirement
                        elif plan_end_pre + survey_inter + 1 < plan_st:
                            if plan_end - dataset['survey_time_min'] + 1 <= plan_end_pre + survey_inter + 1:
                                plan_st = plan_end_pre + survey_inter + 1
                                plan_end = plan_st + dataset['survey_time_min'] - 1
                            else:
                                plan_st = plan_end - dataset['survey_time_min'] + 1

                    # check whether the strip will exceed the maximum lasting time of one boot-up
                    if plan_end - plan0 + 1 <= dataset['boot_time_max']:
                        # check whether the strip will exceed the maximum lasting time of survey in one boot-up
                        if survey['time'] + plan_end - plan_st + 1 <= dataset['survey_boot_time_max']:
                            # Recording the strip
                            survey['strip'].append([plan_st, plan_end, plan_wave])
                            survey['number'] += 1
                            survey['inter'].append(plan_st - plan_end_pre - 1)
                            plan_end_pre = plan_end
                            plan_wave_pre = plan_wave
                            survey['time'] += plan_end - plan_st + 1
                            survey['boot'] += plan_end - plan0 + 1
                            strips_of_orbit[row, col, :] = np.inf
                            continue
                        else:
                            break
                    else:
                        break

                # 任务持续时间大于等于最小持续时间
                else:
                    # 当前条带结束满足最大开机时长的约束
                    if plan_end - plan0 + 1 <= dataset['boot_time_max']:
                        # 满足累计成像时长约束
                        if survey['time'] + plan_end - plan_st + 1 <= dataset['survey_boot_time_max']:
                            # 满足最大成像时长约束
                            if plan_end - plan_st + 1 <= dataset['survey_time_max']:
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['boot'] += plan_end - plan0 + 1
                                strips_of_orbit[row, col, :] = np.inf
                                continue
                            else:
                                plan_end = plan_st + dataset['survey_time_max'] - 1
                                strips_of_orbit[row, col, 0] = plan_end + 1
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['boot'] += plan_end - plan0 + 1
                                continue
                        # 超出累计成像时长约束
                        else:
                            plan_end = plan_st + dataset['survey_boot_time_max'] - survey['time'] - 1
                            strips_of_orbit[row, col, 0] = plan_end + 1
                            # 满足最大成像时长约束
                            if plan_end - plan_st + 1 <= dataset['survey_time_max']:
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                continue
                            else:
                                plan_end = plan_st + dataset['survey_time_max'] - 1
                                strips_of_orbit[row, col, 0] = plan_end + 1
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['boot'] += plan_end - plan0 + 1
                                continue

                    # 部分满足最大开机时长
                    elif plan_end - plan0 + 1 > dataset['boot_time_max'] >= plan_st - plan0:
                        plan_end = plan0 + dataset['boot_time_max'] - 1
                        strips_of_orbit[row, col, 0] = plan_end + 1
                        # 满足累计成像时长约束
                        if survey['time'] + plan_end - plan_st + 1 <= dataset['survey_boot_time_max']:
                            # 满足最大成像时长约束
                            if plan_end - plan_st + 1 <= dataset['survey_time_max']:
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['boot'] += plan_end - plan0 + 1
                                continue
                            else:
                                plan_end = plan_st + dataset['survey_time_max'] - 1
                                strips_of_orbit[row, col, 0] = plan_end + 1
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['boot'] += plan_end - plan0 + 1
                                continue

                        else:
                            plan_end = plan_st + dataset['survey_boot_time_max'] - survey['time'] - 1
                            strips_of_orbit[row, col, 0] = plan_end + 1
                            # 满足最大成像时长约束
                            if plan_end - plan_st + 1 <= dataset['survey_time_max']:
                                # 记录条带
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['boot'] += plan_end - plan0 + 1
                                continue
                            else:
                                plan_end = plan_st + dataset['survey_time_max'] - 1
                                # 记录条带
                                strips_of_orbit[row, col, 0] = plan_end + 1
                                survey['strip'].append([plan_st, plan_end, plan_wave])
                                survey['number'] += 1
                                survey['inter'].append(plan_st - plan_end_pre - 1)
                                plan_end_pre = plan_end
                                plan_wave_pre = plan_wave
                                survey['time'] += plan_end - plan_st + 1
                                survey['boot'] += plan_end - plan0 + 1
                                continue

                    # 不满足最大开机时长
                    else:
                        break

            elif survey['number'] == 4:
                margin = dataset['survey_boot_time_max'] - survey['time']
                index = [i for i in range(len(survey['inter'])) and 0 <= survey['inter'][i] < margin]
                inter = []
                for i in index:
                    if survey['strip'][i - 1][2] == survey['strip'][i][2] and survey['strip'][i][1] - \
                            survey['strip'][i - 1][0] + 1 <= dataset['survey_time_max']:
                        inter.append(survey['inter'][i])

                if not len(inter):
                    id_ = [i for i in range(len(survey['inter'])) and survey['inter'][i] == min(inter)]
                    id_result = []
                    for i in id_:
                        if survey['strip'][i - 1][2] == survey['strip'][i][2] and survey['strip'][i][1] - \
                                survey['strip'][i - 1][0] + 1 <= dataset['survey_time_max']:
                            id_result.append(i)
                    idm = id_result[0]
                    plan_st = survey['strip'][idm - 1][0]
                    plan_end = survey['strip'][idm][1]
                    survey['strip'][idm - 1] = [plan_st, plan_end, plan_wave]
                    survey['strip'][idm] = []
                    survey['number'] -= 1
                    survey['time'] += survey['inter'][idm]
                    survey['inter'][idm] = []
                    continue

            else:
                break

        else:
            break

    return survey, strips_of_orbit


def boot_orbit_build(survey_total_array: np.ndarray, dataset: dict) -> tuple[dict, np.ndarray]:
    """
    Function that plans the boot-ups.
    
    Args:
        survey_total_array: 2D array, the size is (number of surveys, 6), containing the surveys in the given orbit.
        dataset: dict, all the data needed for task allocation, in this function, mainly use the constraints for boot-ups.
            dataset['boot_num_max']: int, the maximum times of boot-ups in each orbit.
            dataset['boot_orbit_time_max']: int, the maximum cumulative time of boot-ups on in each orbit.
            dataset['survey_orbit_num_max']: int, the maximum number of surveys in each orbit.
            dataset['survey_orbit_time_max']: int, the maximum cumulative time of survey in each orbit.
            dataset['boot_time_max']: int, the maximum lasting time of each boot-up.
            dataset['boot_time_min']: int, the minimum lasting time of each boot-up.
            dataset['boot_inter']: int, the minimum interval between two boot-ups.
            dataset['survey_boot_num_max']: int, the maximum number of surveys in each boot-up.
            dataset['survey_boot_time_max']: int, the maximum cumulative time of surveys in each boot-up.
            dataset['survey_time_max']: int, the maximum lasting time of each survey.
            dataset['survey_time_min']: int, the minimum lasting time of each survey.
            dataset['survey_inter_same']: int, the minimum interval between two surveys that are in the same wave.
            dataset['survey_inter_diff']: int, the minimum interval between two surveys that are in different waves.
    """

    boot_build = {'strip': [], 'boot_num': 0, 'survey_num': 0, 'inter': [], 'boot_time': 0, 'survey_time': 0}
    boot_end_pre = 0

    while not np.isinf(survey_total_array).all():
        if boot_build['boot_num'] == 0:
            index_c = [i for i in range(survey_total_array.shape[0]) if survey_total_array[i, 1] == min(survey_total_array[:, 1])][0]
            boot_end_pre = survey_total_array[index_c, 2]

        else:
            if boot_build['boot_num'] < dataset['boot_num_max']:
                index_c = [i for i in range(survey_total_array.shape[0]) if survey_total_array[i, 1] >= boot_end_pre + dataset['boot_inter'] + 1 and survey_total_array[i, 3] <= dataset['survey_orbit_num_max'] - boot_build['survey_num'] and survey_total_array[i, 4] <= dataset['survey_orbit_time_max'] - boot_build['survey_time']]
            else:
                break

        # index_c can be an int or a list, only breaks the loop if index_c is an empty list
        if np.array(index_c).size:
            if type(index_c) is int:
                list_index_c = [index_c]
            else:
                list_index_c = index_c
            array_temp = []
            for i in range(len(list_index_c)):
                boot_st = survey_total_array[list_index_c[i], 1]
                boot_end = survey_total_array[list_index_c[i], 2]
                boot_time = survey_total_array[list_index_c[i], 5]
                if boot_time < dataset['boot_time_min']:
                    if boot_build['boot_num'] == 0:
                        boot_end = boot_st + dataset['boot_time_min'] - 1
                    else:
                        if boot_end_pre + dataset['boot_inter'] + 1 == boot_st:
                            boot_end = boot_st + dataset['boot_time_min'] - 1
                        elif boot_end_pre + dataset['boot_inter'] + 1 < boot_st:
                            if boot_end - dataset['boot_time_min'] + 1 <= boot_end_pre + dataset['boot_inter']:
                                boot_st = boot_end_pre + dataset['boot_inter'] + 1
                                boot_end = boot_st + dataset['boot_time_min'] - 1
                            else:
                                boot_st = boot_end - dataset['boot_time_min'] + 1
                    array_temp.append([survey_total_array[list_index_c[i], 0], boot_st, boot_end, survey_total_array[list_index_c[i], 3], survey_total_array[list_index_c[i], 4], dataset['boot_time_min']])

                else:
                    array_temp.append(list(survey_total_array[list_index_c[i]]))

            index_m = [i for i in range(len(array_temp)) if array_temp[i][5] <= dataset['boot_orbit_time_max'] - boot_build['boot_time']]
            if not len(index_m):
                break

            index_f = [i for i in index_m if array_temp[i][1] == min([array_temp[j][1] for j in index_m])][0]
            no = int(array_temp[index_f][0])
            boot_build['strip'].append(tuple(array_temp[index_f]))
            boot_build['boot_num'] += 1
            boot_build['survey_num'] += array_temp[index_f][3]
            boot_build['inter'].append(array_temp[index_f][1] - boot_end_pre)
            boot_end_pre = array_temp[index_f][2]
            boot_build['boot_time'] += array_temp[index_f][5]
            boot_build['survey_time'] += array_temp[index_f][4]
            survey_total_array[no] = np.inf
            continue
        else:
            break

    return boot_build, survey_total_array


def check_constraints(orbit_plan: np.ndarray, dataset: dict) -> bool:
    """
    Check if the plan for the surveys go in line with the constraints.

    Args:
        orbit_plan: np.ndarray, the plan for the surveys.
        dataset: dict, all the data needed for task allocation, in this function, mainly use the constraints for survey.

    Returns:
        bool, True if the plan goes in line with the constraints, False otherwise.

    """

    orbit_num, cycle_num = orbit_plan.shape

    for orbit in range(orbit_num):
        for cycle in range(cycle_num):
            if type(orbit_plan[orbit, cycle]) is not list:
                if orbit_plan[orbit, cycle].shape[0] > dataset['boot_num_max']:
                    return False
                for i in range(orbit_plan[orbit, cycle].shape[0]):
                    if orbit_plan[orbit, cycle][i, 3] == -1:
                        continue
                    if orbit_plan[orbit, cycle][i, 2] - orbit_plan[orbit, cycle][i, 1] + 1 > dataset['boot_time_max'] or orbit_plan[orbit, cycle][i, 2] - orbit_plan[orbit, cycle][i, 1] + 1 < dataset['boot_time_min']:
                        print(orbit, cycle, i)
                        print(orbit_plan[orbit, cycle][i])
                        return False
                    if i != 0:
                        if orbit_plan[orbit, cycle][i, 1] - orbit_plan[orbit, cycle][i - 1, 2] - 1 <dataset['boot_inter']:
                            return False

    return True


def evaluator(dataset: dict) -> float:
    """
    Function to evaluate the performance of the generated algorithm. Basically we run the task allocation algorithm and compare the results with that of the original algorithm.

    Args:
        dataset: dict, all the data needed for task allocation.

    Returns:
        score: float, shows the performance of the generated algorithm compared with the original one. The higher the score, the better the performance.
            The comparison is conducted based on the total number of cycles and the total time taken. Also, because the reduction of the number of cycles is more important than the reduction of the total time taken, the number of cycles is weighted by 0.7 and the total time taken is weighted by 0.3.

    """

    cycle_num, total_time, orbit_plan = main(dataset)

    # if the planning is in valid
    if cycle_num == 0:
        return -10000

    # if the plan breaks the constraints
    if not check_constraints(orbit_plan, dataset):
        return -10000

    score = -((cycle_num - 30) * 70 + (total_time - 44936) * 30)

    return score


dataset = {
    'task_name': "中国区域",
    'grid_path': "任务栅格数据\\工作模式2_8个波位\\",
    'boot_num_max': 6,
    'boot_orbit_time_max': 500,
    'survey_orbit_num_max': 12,
    'survey_orbit_time_max': 300,
    'boot_time_max': 180,
    'boot_time_min': 10,
    'boot_inter': 60,
    'survey_boot_num_max': 4,
    'survey_boot_time_max': 150,
    'survey_time_max': 60,
    'survey_time_min': 10,
    'survey_inter_same': 26,
    'survey_inter_diff': 52
}

if __name__ == "__main__":
    main(dataset)
    print(evaluator(dataset))
