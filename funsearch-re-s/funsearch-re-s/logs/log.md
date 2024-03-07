# FunSearch 工作日志

## 3月6日

重新设置了24小时仿真时间的基础值，用gpt-4跑了20轮，结果保存在`24hr-20-gpt4-030620.txt`，其中比较有意思的是

```
    """Improved version of `priority_v0`."""
    for i in range(len(sorted_combinations)-1, -1, -1):
        # get the current combination
        curr_combination = sorted_combinations[i]
        # get the current task's point
        curr_points = curr_combination[1]
        # get the current task's satellite, pass, and task index
        curr_task = curr_combination[0]

        # check if current task has the same points with the next task
        # if it is, then we prioritize tasks with lower satellite, pass, and task index
        if i != 0 and curr_points == sorted_combinations[i-1][1]:
            # if current task's indices are lower than the next task's, then we prioritize it
            if curr_task < sorted_combinations[i-1][0]:
                return sorted_combinations.pop(i)

        # if current task doesn't have the same points with the next task
        # or it is the last task (with highest priority), we prioritize it
        else:
            return sorted_combinations.pop(i)

    # return the last task if all tasks have been iterated through
    return sorted_combinations.pop(0)
```

这个函数最后的评分有46.05，比基础值45.63有所优化，在应用于12小时仿真数据上同样有优化，达到了27.05（对比贪心算法26.97），但是在6小时仿真数据上与贪心算法没有差异。

认为是6小时数据集还是太小，可能性太少，体现不出差距。

有意思的是这个算法详细展示了如何走贪心的另一条路，即对于覆盖能力相同的任务，优先安排编号较低的卫星、轨数和任务，根据它的注释，它是想比较具体覆盖的点，而不是单纯的个数，应在这方面考虑，建立一个类似`bin_packing_utils.py`的数据库。

成功建立起了一个基于24hr仿真数据的dataset，节选一点

`
datasets = {
    (2, 6, 7): (22, 23, 24, 86),
    (2, 7, 0): (0, 1, 2, 64),
    (2, 15, 13): (1237, 1238, 1294, 1295),
    (3, 4, 0): (1524, 1525, 1526, 1527)
}
`

差不多还是用三个index定义一个任务，后面的值是可以覆盖的点的index，更新了specification，能跑通，规划时间0.05秒，评估时间0.07秒（我真的觉得很难有比这个再快的了吧）。

应用到funsearch上的specification还没更新，明天起来弄一下。后续我觉得可以多建立起这样的数据集。

另外，我尝试给funsearch添加保存txt运行记录和matplotlib绘制优化过程的代码，但没有效果，我现在还是手动保存运行记录，有点麻烦。比方说我加了这段，但最后啥也没有。

```
if not results[0]:
    log = open('logs/log-24hr.txt', 'a')
    log.write(f'================= Evaluated Program =================\n')
    log.write(f'{function_}\n')
    log.write(f'-----------------------------------------------------\n')
    log.write(f'Score: {str(results)}\n')
    log.write(f'=====================================================\n')
    log.write(f'\n\n')
    log.close()
```

## 3月7日

尝试运行api程序报了numba错

```
This error may have been caused by the following argument(s):
- argument 0: Cannot determine Numba type of <class 'dict'>
```

收集了新的数据集，等api程序更新好之后就可以放进dataset中，最终我想要建设一个类似`bin_packing_utils.py`的数据库，在每个数据集前标注好grid point个数和pass num，这样就不用再传数据集路径作为参数了，只需要dataset即可。