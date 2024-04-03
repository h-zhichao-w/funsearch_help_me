import scipy.io as sio
import numpy as np
from Point2Strip import point2strip
from Strip2StripLX import strip2stripLX

taskFileName = "中国区域"
gridFileName = "任务栅格数据\\工作模式2_8个波位\\"
resultFileName = "贪婪规划结果\\工作模式2_8个波位\\"    # 输出位置

# 加载栅格数据
gridData = sio.loadmat(gridFileName + taskFileName + '.mat')
gridData = np.array(gridData['gridData'])

print(type(gridData))

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
print(gridDataLX[9, 2])
print(cellStripLX[9, 2])
