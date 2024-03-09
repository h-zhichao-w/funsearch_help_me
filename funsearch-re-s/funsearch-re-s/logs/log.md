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

在建立了一个三层字典后，还是没能实现，现在仍然停留在传最低一级的字典的层面上，没有找到在哪里改。

就目前跑下来的结果来看（我也上传了几个记录），能有比贪心高的分数，但是都高的不多，基本上在0.1左右，考虑到现在用的数据集总共包含2836个点，可以理解为就多覆盖了2个或3个网格，优势实在是不明显。

在更新完东南亚数据后跑了一轮，有提升（61.2->62.5）,手动测试了中国数据集也有提升（45.8->45.9）,还是那句话，有提升，但不明显。我把那个函数贴在下面。
```angular2html
# getting the length of coverage of the last prioritised task in the sorted list as a reference point
    reference_length = len(sorted_combinations[list(sorted_combinations.keys())[-1]])

    # creating a shortlisted_tasks list to store tasks with coverage length equal to the reference_length
    shortlisted_tasks = []

    for task, coverage in sorted_combinations.items():
        if len(coverage) == reference_length:
            shortlisted_tasks.append(task)

    # sort the shortlisted tasks in ascending order of their id (i.e. task[2])
    shortlisted_tasks.sort(key=lambda x:x[2])

    # return the first task in the sorted shortlisted tasks
    return shortlisted_tasks[0]
```

## 3月8日

目前采用的方法还是把每个数据集的baseline也写进去，将生成的算法能达到的覆盖率和baseline的覆盖率积做对比，保存这个差值，最后求平均返回。

但是这样会有一个小问题就是，在绘图的时候看不出来到底是没优化（返回值是0）还是是None，而且就算返回值是正数，也不代表对两个数据集都有优化。

00-11-45跑的那次结果是比较理想的，有高出baseline的输出。

我突然又想到，会不会是我们框架定的太死了，把funsearch框在这个贪心里面了，他其实一直只在贪心的这个空间里面找，没跳出去过，所以我做了两次测试。

第一次我把priority函数的返回改成直接返回键值最小的那一个，第二次我写成了return 0让他自己瞎猜了直接，结果发现，首先，FunSearch仍然能优化，而且画出来图特别好看（x，看起来优化效果非常显著（x，其次，FunSearch仍然在朝着贪心算法的方向发展，我尝试了几个得分高的选项，与之前类似，对比贪心算法能有提升但很小。

## 3月9日

我继续沿着昨晚的思路又试了一次，结果还是挺理想的，可以看09-24-19那个记录。这次得分最高的算法我贴在下面：
```angular2html
if not sorted_combinations:
        raise ValueError("No tasks are available for selection.")
    
    # Define weight coefficients for each factor
    coverage_weight = 0.5
    variety_weight = 0.3
    difficulty_weight = 0.2
    
    # Placeholder for the best task and its score
    best_task = None
    best_score = -1
    
    for task_key, coverage_points in sorted_combinations.items():
        # Calculate coverage score
        coverage_score = len(coverage_points)
        
        # Calculate variety score (assuming a hypothetical variety score for simplification)
        variety_score = len(set(coverage_points)) * 2  # Example of emphasizing variety
        
        # Simulate task difficulty score (e.g., based on some external conditions or task characteristics)
        difficulty_score = 1 / (task_key[2] + 1)  # Simplified; assuming higher task indices are more difficult
        
        # Calculate overall task score using weighted sum
        task_score = (coverage_score * coverage_weight +
                      variety_score * variety_weight +
                      difficulty_score * difficulty_weight)
        
        # Select the task if it has the highest score so far
        if task_score > best_score:
            best_score = task_score
            best_task = task_key
    
    if best_task is None:
        raise ValueError("Failed to select a task based on the given criteria.")
    
    # Return the best task key
    return best_task
```
FunSearch给出了一个task score的概念，这个分数有一堆因素构成，我感觉对比我们之前的贪心算是一种不一样的算法？我在CHN和SEA两个数据集上手动测试过了，都有提升（45.8->45.9，61.2->62.5），我觉得这个思路有继续研究的价值，我想增加迭代次数试试看。