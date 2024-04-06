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

print(mission_plan(cellStripLX, survey_time_min=surveyTimeMin, open_time_min=openTimeMin, open_num_max=openNumMax,
                   open_orbit_time_max=openOrbitTimeMax, survey_orbit_num_max=surveyOrbitNumMax,
                   survey_orbit_time_max=surveyOrbitTimeMax, open_time_max=openTimeMax, open_inter=openInter,
                   survey_open_num_max=surveyOpenNumMax, survey_open_time_max=surveyOpenTimeMax,
                   survey_time_max=surveyTimeMax, survey_inter_same=surveyInterSame, survey_inter_diff=surveyInterDiff))
