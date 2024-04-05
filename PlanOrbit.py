import numpy as np
from SurveyOpenBuild import survey_open_build


def plan_orbit(strips, **kwargs):
    """
    Args:
        strips: 2D array, 其大小为 orbitNum 行 waveNum 列，其中 orbitNum 表示轨道数，waveNum 表示波位数，存储有每一轨每个波位的条带。
        **kwargs: dict, 其中包括了规划轨道和成像任务的约束条件。
    """
    k = 0
    survey_total = []

    while True:
        survey, strips = survey_open_build(strips, **kwargs)
        if len(survey['strip']):
            k += 1
            survey['start'] = survey['strip'][0][0]
            survey['end'] = survey['strip'][survey['number'] - 1][1]
            survey_total.append(survey)
        else:
            break

    return survey_total
