import numpy as np

def plan_orbit(strips, **kwargs):
    """
    Args:
        strips: 2D array, 其大小为 orbitNum 行 waveNum 列，其中 orbitNum 表示轨道数，waveNum 表示波位数，存储有每一轨每个波位的条带。
        **kwargs: dict, 其中包括了规划轨道和成像任务的约束条件。
    """
    orbit_num, wave_num = strips.shape
    while True:

