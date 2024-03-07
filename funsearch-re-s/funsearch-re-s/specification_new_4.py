"""
Repositories for solving the task allocation problem.
"""
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from math import sqrt, radians
import copy
import time


def get_grid_data(path: str) -> int:
    """
    Get the number of the grid points from the given path.

    Args:
        path: str, path to the grid data.

    Returns:
        int, number of the grid points.

    """
    grid_data = pd.read_csv(path + 'grid point.csv')
    grid_data = grid_data[['Latitude (deg)', 'Longitude (deg)', 'Area (km^2)']]
    grid_data = grid_data.sort_values(by=['Latitude (deg)', 'Longitude (deg)'], ascending=True)
    grid_data = np.array(grid_data)
    return grid_data.shape[0]


@funsearch.evolve
def priority(sorted_combinations: dict) -> tuple[int, int, int]:
    """
    Get the combination of task id and its coverage that that should be prioritised from the sorted combinations,
    and remove it from the list.
    The function that determine the priority should be generated by LLM

    Args:
        sorted_combinations: dict, sorted combinations of task id and its coverage. The key is a tuple, in the format of [index of satellite, index of pass, index of task], and the value is a tuple, containing the indices of points that can be covered by the task. The tuple is likely to be empty, and if so, it means that the corresponding task cannot cover any point. The dict is sorted in ascending order on the number of points that can be covered.

    Returns:
        tuple[int, int, int], the combination that is prioritised,
        in the format of [satellite index, pass index, task index], i.e. the key of the prioritised task.

    """

    return list(sorted_combinations.keys())[-1]


def update_total_coverage(selected_task: tuple, sorted_combination: dict) -> dict:
    """
    Update the total coverage based on the selected task, and remove the selected task from the sorted combinations.

    Args:
        selected_task: tuple[int, int, int], the selected task that should be removed from the sorted combinations, and the points that has been covered by it should be removed from other remaining tasks.
        sorted_combination: dict, sorted combinations of task id and its coverage. The key is a tuple, in the format of [index of satellite, index of pass, index of task], and the value is a tuple, containing the indices of points that can be covered by the task. The tuple is likely to be empty, and if so, it means that the corresponding task cannot cover any point. The dict is sorted in ascending order on the number of points that can be covered.

    Returns:
        dict, updated total coverage based on the selected task.

    """
    newly_covered_points = sorted_combination[selected_task]
    del sorted_combination[selected_task]
    for key, value in sorted_combination.items():
        sorted_combination[key] = tuple(point for point in value if point not in newly_covered_points)
    sorted_combination = dict(sorted(sorted_combination.items(), key=lambda item: len(item[1])))

    return sorted_combination


def main(sat_num: int, pass_num: list, sorted_combinations: dict, constraints: list) -> list:
    """
    Main function for task allocation.

    Args:
        sat_num: int, number of the satellites in the system.
        pass_num: list, number of the passes for each satellite.
        sorted_combinations: dict, sorted combinations of task id and its coverage. The key is a tuple, in the format of [index of satellite, index of pass, index of task], and the value is a tuple, containing the indices of points that can be covered by the task. The tuple is likely to be empty, and if so, it means that the corresponding task cannot cover any point. The dict is sorted in ascending order on the number of points that can be covered.
        constraints: list, constraints for task allocation, containing maximum accumulated working time
        per pass, maximum working time per image, minimum working time per image, and maximum image per pass.

    Returns:
        list, selected tasks for each satellite in each pass.

    """
    max_time_per_pass = constraints[0]
    max_time_per_image = constraints[1]
    min_time_per_image = constraints[2]
    max_image_per_pass = constraints[3]

    # Initialization
    final_schedule = [[] for _ in range(sat_num)]  # Record assigned tasks
    mission_time_tracker = [[] for _ in range(sat_num)]  # Record the total time of assigned tasks
    for i in range(sat_num):
        final_schedule[i] = [[] for _ in range(pass_num[i])]

    # Greedy algorithm
    covered_points = set()  # Record the points that have been covered
    while sorted_combinations:
        (sat_idx, pass_idx, task_idx) = priority(sorted_combinations)
        if len(sorted_combinations[(sat_idx, pass_idx, task_idx)]) <= 0:
            break
        task_points = sorted_combinations[(sat_idx, pass_idx, task_idx)]
        # if this task is able to apply
        if len(final_schedule[sat_idx][pass_idx]) < max_image_per_pass:
            new_covered_points = set(task_points) - covered_points
            # If the task can cover points that have not been covered yet
            if new_covered_points:
                final_schedule[sat_idx][pass_idx].append(task_idx)
                covered_points.update(task_points)
                # update the dict
                sorted_combinations = update_total_coverage(
                    selected_task=(sat_idx, pass_idx, task_idx), sorted_combination=sorted_combinations)
            else:
                del sorted_combinations[(sat_idx, pass_idx, task_idx)]
        else:
            del sorted_combinations[(sat_idx, pass_idx, task_idx)]

    return final_schedule

@funsearch.run
def evaluator(
        sorted_combinations: dict,
        path: str = 'D:\\OneDrive - sjtu.edu.cn\\Bachelor Thesis\\Simulation Data\\Simulation Time-6hr\\',
        sat_num: int = 4,
        pass_num: list = [16, 17, 17, 17],
        constraints: list = [300, 60, 10, 5],
    ) -> float:
    """
    Evaluation function for the algorthm of task allocation.

    Args:
        sorted_combinations: dict, sorted combinations of task id and its coverage. The key is a tuple, in the format of [index of satellite, index of pass, index of task], and the value is a tuple, containing the indices of points that can be covered by the task. The tuple is likely to be empty, and if so, it means that the corresponding task cannot cover any point. The dict is sorted in ascending order on the number of points that can be covered.
        path: str, path to the grid data.
        sat_num: int, number of the satellites in the system.
        pass_num: list, number of the passes for each satellite.
        constraints: list, constraints for task allocation, containing maximum accumulated working time
        per pass, maximum working time per image, minimum working time per image, and maximum image per pass.

    Returns:
        float, evaluation result, which is coverage level in percentage.

    """

    max_time_per_pass = constraints[0]
    max_time_per_image = constraints[1]
    min_time_per_image = constraints[2]
    max_image_per_pass = constraints[3]

    # Initialization
    final_schedule = [[] for _ in range(sat_num)]  # Record assigned tasks
    mission_time_tracker = [[] for _ in range(sat_num)]  # Record the total time of assigned tasks
    for i in range(sat_num):
        final_schedule[i] = [[] for _ in range(pass_num[i])]

    initial_dataset = copy.deepcopy(sorted_combinations)

    # Greedy algorithm
    covered_points = set()  # Record the points that have been covered
    while sorted_combinations:
        (sat_idx, pass_idx, task_idx) = priority(sorted_combinations)
        if len(sorted_combinations[(sat_idx, pass_idx, task_idx)]) <= 0:
            break
        task_points = sorted_combinations[(sat_idx, pass_idx, task_idx)]
        # if this task is able to apply
        if len(final_schedule[sat_idx][pass_idx]) < max_image_per_pass:
            new_covered_points = set(task_points) - covered_points
            # If the task can cover points that have not been covered yet
            if new_covered_points:
                final_schedule[sat_idx][pass_idx].append(task_idx)
                covered_points.update(task_points)
                # update the dict
                sorted_combinations = update_total_coverage(
                    selected_task=(sat_idx, pass_idx, task_idx), sorted_combination=sorted_combinations)
            else:
                del sorted_combinations[(sat_idx, pass_idx, task_idx)]
        else:
            del sorted_combinations[(sat_idx, pass_idx, task_idx)]

    # Evaluate the algorithm
    covered_points = set()  # Record the points that have been covered
    for sat_idx in range(sat_num):
        for pass_idx in range(pass_num[sat_idx]):
            if final_schedule[sat_idx][pass_idx]:
                for task_idx in final_schedule[sat_idx][pass_idx]:
                    task_points = initial_dataset[(sat_idx, pass_idx, task_idx)]
                    covered_points.update(task_points)

    # Number of the points that have been covered
    num_covered_points = len(covered_points)
    # Coverage level in percentage

    num_grid_points = get_grid_data(path)
    coverage_level = num_covered_points / num_grid_points * 100

    return coverage_level
