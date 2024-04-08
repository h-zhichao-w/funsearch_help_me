"""
Specification for task allocation problem. The goal is to allocate tasks to the Strict Return to Orbit (SRO) satellites in a way that minimises the total cycles and time taken to complete all tasks.
"""
# Repositories for solving the task allocation problem.
import scipy.io as sio
import numpy as np
from tqdm import tqdm


def main(dataset: dict) -> tuple[int, int]:
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



#
# orbitPlan, surveyPlan, sj = mission_plan(cellStripLX, survey_time_min=surveyTimeMin, open_time_min=openTimeMin,
#                                          open_num_max=openNumMax,
#                                          open_orbit_time_max=openOrbitTimeMax, survey_orbit_num_max=surveyOrbitNumMax,
#                                          survey_orbit_time_max=surveyOrbitTimeMax, open_time_max=openTimeMax,
#                                          open_inter=openInter,
#                                          survey_open_num_max=surveyOpenNumMax, survey_open_time_max=surveyOpenTimeMax,
#                                          survey_time_max=surveyTimeMax, survey_inter_same=surveyInterSame,
#                                          survey_inter_diff=surveyInterDiff)
# Sno = np.where(sj[:] == 1)
# orbit_num = surveyPlan.size
# col_num = max(len(orbitPlan[i]) for i in range(orbit_num))
# orbitPlan_array = np.empty((orbit_num, col_num), dtype=object)
# orbitPlan_array.fill([])
# for i in range(orbit_num):
#     for j in range(len(orbitPlan[i])):
#         strips = orbitPlan[i][j][0]
#         strips = np.hstack((strips, np.ones((strips.shape[0], 1)) * orbitPlan[i][j][1]))
#         strips = strips.astype(int)
#         orbitPlan_array[i][j] = strips
#
# cycle_num = max(len(surveyPlan[i]) for i in range(orbit_num))
# surveyPlan_array = np.empty((orbit_num, cycle_num), dtype=object)
# surveyPlan_array.fill([])
# for i in range(orbit_num):
#     for j in range(len(surveyPlan[i])):
#         strips = surveyPlan[i][j]
#         strips = strips.astype(int)
#         surveyPlan_array[i][j] = strips
#
# timeTotal = np.zeros(cycle_num)
# for k in range(cycle_num):
#     for i in range(len(Sno[0])):
#         sno = Sno[0][i]
#         if type(surveyPlan_array[sno, k]) is not list:
#             timeTotal[k] += sum(surveyPlan_array[sno, k][:, 1] - surveyPlan_array[sno, k][:, 0] + 1)
#         else:
#             continue
#
# print(timeTotal)
# print(sum(timeTotal))