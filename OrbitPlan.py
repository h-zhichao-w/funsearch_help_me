import scipy.io as sio
import numpy as np
from Point2Strip import point2strip
from Strip2StripLX import strip2stripLX
from MissionPlan import mission_plan

taskFileName = "中国区域"
gridFileName = "任务栅格数据\\工作模式2_8个波位\\"
resultFileName = "贪婪规划结果\\工作模式2_8个波位\\"  # 输出位置

# 加载栅格数据
gridData = sio.loadmat(gridFileName + taskFileName + '.mat')
gridData = np.array(gridData['gridData'])

# 约束条件
openNumMax = 6
openOrbitTimeMax = 500
surveyOrbitNumMax = 12
surveyOrbitTimeMax = 300
openTimeMax = 180
openTimeMin = 10
openInter = 60
surveyOpenNumMax = 4
surveyOpenTimeMax = 150
surveyTimeMax = 60
surveyTimeMin = 10
surveyInterSame = 26
surveyInterDiff = 52

gridDataLX = point2strip(gridData)
cellStripLX = strip2stripLX(gridDataLX, surveyTimeMin, openTimeMin)

orbitPlan, surveyPlan, sj = mission_plan(cellStripLX, survey_time_min=surveyTimeMin, open_time_min=openTimeMin,
                                     open_num_max=openNumMax,
                                     open_orbit_time_max=openOrbitTimeMax, survey_orbit_num_max=surveyOrbitNumMax,
                                     survey_orbit_time_max=surveyOrbitTimeMax, open_time_max=openTimeMax,
                                     open_inter=openInter,
                                     survey_open_num_max=surveyOpenNumMax, survey_open_time_max=surveyOpenTimeMax,
                                     survey_time_max=surveyTimeMax, survey_inter_same=surveyInterSame,
                                     survey_inter_diff=surveyInterDiff)
Sno = np.where(sj[:] == 1)
orbit_num = surveyPlan.size
col_num = max(len(orbitPlan[i]) for i in range(orbit_num))
orbitPlan_array = np.empty((orbit_num, col_num), dtype=object)
orbitPlan_array.fill([])
for i in range(orbit_num):
    for j in range(len(orbitPlan[i])):
        strips = orbitPlan[i][j][0]
        strips = np.hstack((strips, np.ones((strips.shape[0], 1)) * orbitPlan[i][j][1]))
        strips = strips.astype(int)
        orbitPlan_array[i][j] = strips

cycle_num = max(len(surveyPlan[i]) for i in range(orbit_num))
surveyPlan_array = np.empty((orbit_num, cycle_num), dtype=object)
surveyPlan_array.fill([])
for i in range(orbit_num):
    for j in range(len(surveyPlan[i])):
        strips = surveyPlan[i][j]
        strips = strips.astype(int)
        surveyPlan_array[i][j] = strips

timeTotal = np.zeros(cycle_num)
for k in range(cycle_num):
    for i in range(len(Sno[0])):
        sno = Sno[0][i]
        if type(surveyPlan_array[sno, k]) is not list:
            # timeTotal(k) =  timeTotal(k)+sum(surveyPlan{Sno(i),k}(:,2)-surveyPlan{Sno(i),k}(:,1)+1);
            timeTotal[k] += sum(surveyPlan_array[sno, k][:, 1] - surveyPlan_array[sno, k][:, 0] + 1)
        else:
            continue

print(len(timeTotal))
