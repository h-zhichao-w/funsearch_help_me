import numpy as np
from tqdm import tqdm


def point2strip(grid_data):
    """
    Args:
        grid_data: 2D array, 其大小为 orbitNum 行 waveNum 列，其中 orbitNum 表示轨道数，waveNum 表示波位数。
    
    Returns:
        gridDataLX: 经过处理后的栅格数据，仍为二维 cell 数组，大小与输入参数 gridData 相同，LX for 连续。
    """

    orbit_num, wave_num = grid_data.shape[0], grid_data.shape[1]
    gridDataLX = np.zeros((orbit_num, wave_num), dtype=object)

    for orbit in tqdm(range(orbit_num)):
        for wave in range(wave_num):
            if not grid_data[orbit, wave].size:
                continue
            grid_cell = grid_data[orbit, wave]
            pst = 0
            pend = 1
            LX = []
            while True:
                if pend == grid_cell.shape[0]:
                    LX.append([grid_cell[pst, 0], grid_cell[pend - 1, 0], grid_cell[pst, 2]])
                    break
                if grid_cell[pend, 0] - grid_cell[pend - 1, 0] == 1:
                    pend += 1
                    continue
                if grid_cell[pend, 0] - grid_cell[pend - 1, 0] > 1:
                    LX.append([grid_cell[pst, 0], grid_cell[pend - 1, 0], grid_cell[pst, 2]])
                    pst = pend
                    pend += 1
                    continue
            gridDataLX[orbit, wave] = LX

    return gridDataLX
