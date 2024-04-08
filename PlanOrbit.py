import numpy as np
from SurveyOpenBuild import survey_open_build
from OpenOrbitBuild import open_orbit_build


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

    survey_total_array = np.zeros((len(survey_total), 6))
    for i in range(len(survey_total)):
        survey_total_array[i, 0] = i
        survey_total_array[i, 1] = survey_total[i]['start']
        survey_total_array[i, 2] = survey_total[i]['end']
        survey_total_array[i, 3] = survey_total[i]['number']
        survey_total_array[i, 4] = survey_total[i]['time']
        survey_total_array[i, 5] = survey_total_array[i, 2] - survey_total_array[i, 1]

    k = 0
    orbit_total = []
    while True:
        open_build, survey_total_array = open_orbit_build(survey_total_array, **kwargs)
        if open_build['strip']:
            k += 1
            open_build['start'] = open_build['strip'][0][1]
            open_build['end'] = open_build['strip'][open_build['open_num'] - 1][2]
            orbit_total.append(open_build)
        else:
            break

    return survey_total, orbit_total
