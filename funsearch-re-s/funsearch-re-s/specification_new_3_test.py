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


def is_point_in_strip(point, endpoint1, endpoint2, range_distance: float) -> bool:
    """
    Check if a given point is located in a strip defined by two endpoints and a range distance.

    Args:
        point: tuple, latitude and longitude of the point.
        endpoint1: tuple, latitude and longitude of the first endpoint.
        endpoint2: tuple, latitude and longitude of the second endpoint.
        range_distance: float, range distance.

    Returns:
        bool, whether the point is located in the strip.

    """
    mid = (endpoint1 + endpoint2) / 2
    lat, lon = point[0] - mid[0], point[1] - mid[1]
    lat1, lon1 = endpoint1[0] - mid[0], endpoint1[1] - mid[1]
    lat2, lon2 = endpoint2[0] - mid[0], endpoint2[1] - mid[1]

    # Haversine formula to calculate distances
    R = 6371.0  # Earth radius in kilometers
    lat, lon, lat1, lon1, lat2, lon2 = map(radians, [lat, lon, lat1, lon1, lat2, lon2])

    # Calculate the length of the line segment
    line_length = sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

    # Calculate the dot product between the point and the line segment
    dot_product = ((lat - lat1) * (lat2 - lat1) + (lon - lon1) * (lon2 - lon1)) / line_length ** 2

    if 0 <= dot_product <= 1:
        # Calculate the closest point on the line segment to the given point
        closest_lat = lat1 + dot_product * (lat2 - lat1)
        closest_lon = lon1 + dot_product * (lon2 - lon1)

        # Calculate the distance from the point to the closest point on the line
        distance = R * sqrt((lat - closest_lat) ** 2 + (lon - closest_lon) ** 2)

        # Check if the distance is within the specified range
        return distance <= range_distance
    else:
        return min(R * sqrt((lat - lat1) ** 2 + (lon - lon1) ** 2),
                   R * sqrt((lat - lat2) ** 2 + (lon - lon2) ** 2)) <= range_distance


def convert_time_to_seconds(time_stamp: np.ndarray, start_time: int) -> np.ndarray:
    """
    Convert time stamp to seconds.

    Args:
        time_stamp: np.ndarray, an array consists of time stamps.
        start_time: int, global start time.

    Returns:
        np.ndarray, an array consists of time in seconds.
        float, global start time.

    """
    for i in range(time_stamp.shape[0]):
        for j in range(time_stamp.shape[1]):
            t = time.strptime(time_stamp[i, j], "%d %b %Y %H:%M:%S.%f")
            time_stamp[i, j] = time.mktime(t)
    return time_stamp - start_time


def process_time_data(data: pd.DataFrame, start_time: int) -> np.ndarray:
    """
    Extract 'Pass', 'Start Time (UTCG)', and 'End Time (UTCG)' from the given data and convert time stamp to seconds.

    Args:
        data: pd.DataFrame, data to be processed.
        start_time: int, global start time.

    Returns:
        np.ndarray, processed data.
         float, global start time.

    """
    try:
        data_array = np.array(data[['Pass', 'Start Time (UTCG)', 'End Time (UTCG)']])
    except KeyError:
        data_array = np.array(data)
    time_stamp = data_array[:, 1:3]
    time_stamp[:, :] = convert_time_to_seconds(time_stamp, start_time)
    data_array[:, 1:3] = time_stamp
    return data_array


def get_access_data(path: str, sat_num: int, start_time: int) -> list:
    """
    Get access data of all satellites from the given path.

    Args:
        path: str, path to the data without file name.
        sat_num: int, number of the satellites in the system.
        start_time: int, global start time.

    Returns:
        list, access data.

    """
    access_data = []
    for i in range(1, sat_num + 1):
        sat_pass_path = path + 'PassData{}'.format(i) + '.csv'
        access_path = path + 'access{}'.format(i) + '.csv'
        sat_pass = pd.read_csv(sat_pass_path)
        access = pd.read_csv(access_path)

        sat_pass_array = process_time_data(sat_pass, start_time)
        access_array = process_time_data(access, start_time)

        current_access_data = []
        current_access_index: int = 0
        for pass_num in range(sat_pass_array.shape[0]):
            # If all access have been studied, save an empty list
            if current_access_index >= access_array.shape[0]:
                current_access_data.append([np.array([])])

            # If the current access is in the current pass
            elif (access_array[current_access_index, 1] >= sat_pass_array[pass_num, 1]
                  and sat_pass_array[pass_num, 2] >= access_array[current_access_index, 2]):
                duration = np.arange(
                    access_array[current_access_index, 1],
                    access_array[current_access_index, 2] + 1,
                    1
                )
                current_access_data.append([duration])

                # Next access
                current_access_index += 1

            else:
                current_access_data.append([np.array([])])

        access_data.append(current_access_data)

    return access_data


def get_grid_data(path: str) -> np.ndarray:
    """
    Get the latitude and longitude of the grid points from the given path.

    Args:
        path: str, path to the grid data.

    Returns:
        np.ndarray, grid data.

    """
    grid_data = pd.read_csv(path + 'grid point.csv')
    grid_data = grid_data[['Latitude (deg)', 'Longitude (deg)', 'Area (km^2)']]
    grid_data = grid_data.sort_values(by=['Latitude (deg)', 'Longitude (deg)'], ascending=True)
    grid_data = np.array(grid_data)
    return grid_data


def get_sensor_data(path: str, sat_num: int, half_cone_angle: float) -> list:
    """
    Get location of sensors of satellites in latitude, longitude, and range from Earth from the given path, and
    convert the range from Earth into the range of field of vision on the ground.

    Args:
        path: str, path to the sensor data.
        sat_num: int, number of the satellites in the system.
        half_cone_angle: float, half cone angle of the sensor in radius.

    Returns:
        list, sensor data.

    """
    sensor_data = []
    for i in range(1, sat_num + 1):
        sensor_path = path + 'sensor{}'.format(i) + '.csv'
        sensor = pd.read_csv(sensor_path)
        sensor = sensor[['Latitude (deg)', 'Longitude (deg)', 'Range (km)']]
        sensor = np.array(sensor)
        sensor[:, 2] = sensor[:, 2] * np.tan(half_cone_angle)
        sensor_data.append(sensor)
    return sensor_data


def task_division(access_data: list, sat_num: int, max_time_per_image: float, min_time_per_image: float, time_step: int) -> tuple[list, int, list]:
    """
    Divide tasks for each satellite in each pass.

    Args:
        access_data: list, access data recording access time in each pass of each satellite.
        sat_num: int, number of the satellites in the system.
        max_time_per_image: float, maximum time per image.
        min_time_per_image: float, minimum time per image.
        time_step: int, time step for task division.

    Returns:
        list, divided tasks for each satellite in each pass.
        int, number of tasks.
        list, number of passes for each satellite.

    """

    pass_num = [len(sat) for sat in access_data]

    # Initialization
    sat_task = []
    task_num = 0
    for sat_idx in range(sat_num):
        sat_task.append([])

    for sat_idx in range(sat_num):
        for pass_idx in range(pass_num[sat_idx]):
            sat_task[sat_idx].append([])
            if access_data[sat_idx][pass_idx][0].size:  # if there is access in the current pass
                access_time = list(access_data[sat_idx][pass_idx][0])
                while len(access_time):
                    start_time = access_time[0]
                    end_time = start_time + max_time_per_image
                    if end_time > access_time[-1]:
                        end_time = access_time[-1]
                    task = [start_time, end_time]
                    # If this task cannot satisfy the shortest time requirement, the remaining tasks cannot be satisfied neither, thus break the loop
                    if end_time - start_time < min_time_per_image:
                        break
                    sat_task[sat_idx][pass_idx].append(task)
                    task_num += 1
                    # Step forward
                    access_time = access_time[access_time.index(start_time) + time_step:]
            else:
                # if there is no access in the current pass, append an empty list
                sat_task[sat_idx][pass_idx].append([])

    return sat_task, task_num, pass_num


def get_task_cover_point(sat_task: list, sat_num: int, pass_num: list, sensor_data: list, grid_data: np.ndarray) -> tuple[list, dict]:
    """
    Based on the result of task division, calculate the grid points that can be covered by each task.

    Args:
        sat_task: list, divided tasks for each satellite in each pass.
        sat_num: int, number of the satellites in the system.
        pass_num: list, number of passes for each satellite.
        sensor_data: list, location of sensors of satellites.
        grid_data: np.ndarray, latitude and longitude of the grid points.

    Returns:
        list, grid points that can be covered by each task.
        dict, number of total coverage of each task.

    """
    task_cover_point = []
    for i in tqdm(range(sat_num)):
        task_cover_point.append([])
        for j in range(pass_num[i]):
            task_cover_point[i].append([])
            if sat_task[i][j][0]:
                for k in range(len(sat_task[i][j])):
                    task_cover_point[i][j].append([])
                    task = sat_task[i][j][k]
                    start_time = task[0]
                    end_time = task[1]
                    endpoint1 = sensor_data[i][int(start_time)]
                    endpoint2 = sensor_data[i][int(end_time)]
                    range_distance = (sensor_data[i][int(start_time)][2] + sensor_data[i][int(end_time)][
                        2]) / 2
                    for point in grid_data:
                        if is_point_in_strip(point, endpoint1, endpoint2, range_distance):
                            task_cover_point[i][j][k].append(tuple(point))
            else:
                task_cover_point[i][j].append([])

    total_coverage: dict[tuple[int, int, int], int] = {}

    for i in range(sat_num):
        for j in range(pass_num[i]):
            if sat_task[i][j][0]:
                for k in range(len(sat_task[i][j])):
                    total_coverage[(i, j, k)] = len(task_cover_point[i][j][k])
            else:
                total_coverage[(i, j, 0)] = 0

    return task_cover_point, total_coverage

def priority(sorted_combinations: list) -> tuple[tuple[int, int, int], int]:
    """
    Get the combination of task id and its coverage that that should be prioritised from the sorted combinations,
    and remove it from the list.
    The function that determine the priority should be generated by LLM

    Args:
        sorted_combinations: list, sorted combinations of task id and its coverage.

    Returns:
        tuple[tuple[int, int, int], int], the combination that is prioritised,
        in the format of ((satellite index, pass index, task index), number of the points that can be covered).

    """

    max_satellite_index = -1
    max_pass_index = -1
    max_task_index = -1
    max_coverage = -1

    for idx, comb in enumerate(sorted_combinations):
        sat_idx, pass_idx, task_idx = comb[0]
        coverage = comb[1]

        if coverage > max_coverage:
            max_satellite_index = sat_idx
            max_pass_index = pass_idx
            max_task_index = task_idx
            max_coverage = coverage

    if max_satellite_index != -1:
        for idx, comb in enumerate(sorted_combinations):
            sat_idx, pass_idx, task_idx = comb[0]
            coverage = comb[1]
            if sat_idx == max_satellite_index and pass_idx == max_pass_index and task_idx == max_task_index:
                return sorted_combinations.pop(idx)

def update_total_coverage(sat_num: int, pass_num: list, task_cover_point: list, new_covered_points: set) -> dict:
    """
    Update the total coverage based on the updated set of covered points

    Args:
        sat_num: int, number of the satellites in the system.
        pass_num: list, number of passes for each satellite.
        task_cover_point: list, grid points that can be covered by each task originally.
        new_covered_points: set, newly covered points by the selected task.

    Returns:
        dict, updated total coverage.

    """
    total_coverage = {}
    for i in range(sat_num):
        for j in range(pass_num[i]):
            for k, points in enumerate(task_cover_point[i][j]):
                task_cover_point[i][j][k] = [p for p in points if p not in new_covered_points]
                total_coverage[(i, j, k)] = len(task_cover_point[i][j][k])
    return total_coverage



def main(path: str, sat_num: int, half_cone_angle: float, constraints: list, time_step: int, start_time: int) -> list:
    """
    Main function for task allocation.

    Args:
        path: str, path to the data without file name.
        sat_num: int, number of the satellites in the system.
        half_cone_angle: float, half cone angle of the sensor in radius.
        constraints: list, constraints for task allocation, containing maximum accumulated working time
        per pass, maximum working time per image, minimum working time per image, and maximum image per pass.
        time_step: int, time step for task division.
        start_time: int, global start time.

    Returns:
        list, selected tasks for each satellite in each pass.

    """
    max_time_per_pass = constraints[0]
    max_time_per_image = constraints[1]
    min_time_per_image = constraints[2]
    max_image_per_pass = constraints[3]

    access_data = get_access_data(path, sat_num, start_time)
    grid_data = get_grid_data(path)
    sensor_data = get_sensor_data(path, sat_num, half_cone_angle)

    sat_task, task_num, pass_num = task_division(access_data, sat_num, max_time_per_image, min_time_per_image, time_step)
    task_cover_point, total_coverage = get_task_cover_point(sat_task, sat_num, pass_num, sensor_data, grid_data)

    sorted_combinations = sorted(total_coverage.items(), key=lambda x: x[1])
    print(sorted_combinations)
    # Initialization
    final_schedule = [[] for _ in range(sat_num)]  # Record assigned tasks
    mission_time_tracker = [[] for _ in range(sat_num)]  # Record the total time of assigned tasks
    for i in range(sat_num):
        mission_time_tracker[i] = [0] * pass_num[i]
        final_schedule[i] = [[] for _ in range(pass_num[i])]

    # Greedy algorithm
    covered_points = set()  # Record the points that have been covered
    while sorted_combinations:
        (sat_idx, pass_idx, task_idx), coverage = priority(sorted_combinations)
        if coverage <= 0:
            break
        task_points = task_cover_point[sat_idx][pass_idx][task_idx]
        task_time = sat_task[sat_idx][pass_idx][task_idx][1] - sat_task[sat_idx][pass_idx][task_idx][0]
        # if this task is able to apply
        if (len(final_schedule[sat_idx][pass_idx]) < max_image_per_pass) and (
                mission_time_tracker[sat_idx][pass_idx] + task_time <= max_time_per_pass):
            new_covered_points = set(task_points) - covered_points
            # If the task can cover points that have not been covered yet
            if new_covered_points:
                final_schedule[sat_idx][pass_idx].append(task_idx)
                mission_time_tracker[sat_idx][pass_idx] += task_time
                covered_points.update(task_points)
                # Remove the task that has been arranged
                task_cover_point[sat_idx][pass_idx][task_idx] = []
                # Remove covered points from remaining tasks,
                # and recalculate total_coverage based on the updated set of uncovered points
                total_coverage = update_total_coverage(sat_num, pass_num, task_cover_point, new_covered_points)
                # Re-sort combinations based on the updated total_coverage
                sorted_combinations = sorted(total_coverage.items(), key=lambda x: x[1])

    return final_schedule

def evaluator(
        path: str = '/home/jty/Code/zhengkan/deepmind/spe_exp/Simulation+Time-6hr/Simulation+Time-6hr/',
        sat_num: int = 4,
        half_cone_angle: float = 20 * np.pi / 180,
        constraints: list = [300, 60, 10, 5],
        time_step: int = 60,
        start_time: int = 1706342400) -> float:
    """
    Evaluation function for the algorthm of task allocation.

    Args:
        path: str, path to the data without file name.
        sat_num: int, number of the satellites in the system.
        half_cone_angle: float, half cone angle of the sensor in radius.
        constraints: list, constraints for task allocation, containing maximum accumulated working time
        per pass, maximum working time per image, minimum working time per image, and maximum image per pass.
        time_step: int, time step for task division.
        start_time: int, global start time.

    Returns:
        float, evaluation result, which is coverage level in percentage.

    """

    max_time_per_pass = constraints[0]
    max_time_per_image = constraints[1]
    min_time_per_image = constraints[2]
    max_image_per_pass = constraints[3]

    access_data = get_access_data(path, sat_num, start_time)
    grid_data = get_grid_data(path)
    sensor_data = get_sensor_data(path, sat_num, half_cone_angle)

    sat_task, task_num, pass_num = task_division(access_data, sat_num, max_time_per_image, min_time_per_image,
                                                 time_step)
    task_cover_point, total_coverage = get_task_cover_point(sat_task, sat_num, pass_num, sensor_data, grid_data)
    copy_task_cover_point = copy.deepcopy(task_cover_point)

    sorted_combinations = sorted(total_coverage.items(), key=lambda x: x[1])

    # Initialization
    final_schedule = [[] for _ in range(sat_num)]  # Record assigned tasks
    mission_time_tracker = [[] for _ in range(sat_num)]  # Record the total time of assigned tasks
    for i in range(sat_num):
        mission_time_tracker[i] = [0] * pass_num[i]
        final_schedule[i] = [[] for _ in range(pass_num[i])]

    # Generate arrangement
    covered_points = set()  # Record the points that have been covered
    while sorted_combinations:
        (sat_idx, pass_idx, task_idx), coverage = priority(sorted_combinations)
        if coverage <= 0:
            break
        task_points = task_cover_point[sat_idx][pass_idx][task_idx]
        task_time = sat_task[sat_idx][pass_idx][task_idx][1] - sat_task[sat_idx][pass_idx][task_idx][0]
        # if this task is able to apply
        if (len(final_schedule[sat_idx][pass_idx]) < max_image_per_pass) and (
                mission_time_tracker[sat_idx][pass_idx] + task_time <= max_time_per_pass):
            new_covered_points = set(task_points) - covered_points
            # If the task can cover points that have not been covered yet
            if new_covered_points:
                final_schedule[sat_idx][pass_idx].append(task_idx)
                mission_time_tracker[sat_idx][pass_idx] += task_time
                covered_points.update(task_points)
                # Remove the task that has been arranged
                task_cover_point[sat_idx][pass_idx][task_idx] = []
                # Remove covered points from remaining tasks,
                # and recalculate total_coverage based on the updated set of uncovered points
                total_coverage = update_total_coverage(sat_num, pass_num, task_cover_point, new_covered_points)
                # Re-sort combinations based on the updated total_coverage
                sorted_combinations = sorted(total_coverage.items(), key=lambda x: x[1])

    # Evaluate the algorithm
    covered_points = set()  # Record the points that have been covered
    for sat_idx in range(sat_num):
        for pass_idx in range(pass_num[sat_idx]):
            if final_schedule[sat_idx][pass_idx]:
                for task_idx in final_schedule[sat_idx][pass_idx]:
                    task_points = copy_task_cover_point[sat_idx][pass_idx][task_idx]
                    covered_points.update(task_points)

    # Number of the points that have been covered
    num_covered_points = len(covered_points)
    # Coverage level in percentage
    coverage_level = num_covered_points / grid_data.shape[0] * 100

    return coverage_level


def get_start_time(
        time_text: str = '2024/1/27  16:00:00',
        format: str = '%Y/%m/%d %H:%M:%S') -> int:
    """
    Convert time in plain text to seconds (int).

    Args:
        time_text: str, plain text of time
        format: str, format of the time.

    Returns:
        int, time in seconds

    """
    time_array = time.strptime(time_text, format)
    start_time = time.mktime(time_array)
    return int(start_time)

start_time_24hr = 1709661600
start_time_12hr = 1705428000
start_time_6hr = 1706360400  # 这个start_time好像和时区有关，我在上海设置好了之后来莫斯科跑就会报错，这个是我在莫斯科重新设置的，如果报错的话可能要你重新弄一下，我留了一个函数，用默认参数可以直接跑
start_time_3hr = 1706446800
constraints = [300, 60, 10, 5]
time_step = 60
half_cone_angle = 20 * np.pi / 180
sat_num = 4

if __name__ == '__main__':
    print(get_start_time('2024/1/16  21:00:00'))
    sat_task_selected = main(path='D:\\OneDrive - sjtu.edu.cn\\Bachelor Thesis\\Simulation Data\\Simulation Time-6hr\\', sat_num=sat_num, half_cone_angle=half_cone_angle, constraints=constraints, time_step=time_step, start_time=start_time_6hr)
    print(sat_task_selected)
    coverage_level = evaluator(path='D:\\OneDrive - sjtu.edu.cn\\Bachelor Thesis\\Simulation Data\\Simulation Time-6hr\\', sat_num=sat_num, half_cone_angle=half_cone_angle, constraints=constraints, time_step=time_step, start_time=start_time_6hr)
    print(coverage_level)
