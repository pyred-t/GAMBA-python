# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import numpy as np

import GAMBA as gm


def print_hi():
    f = gm.load_example_coregistered()
    res_Y = gm.group_regions(f)
    ctx_l = np.where(np.char.find(res_Y['regionDescriptions'], 'ctx-lh-') != -1)

    img_data = res_Y['data'][ctx_l]
    res = gm.association(img_data, ['APOE', 'APP', 'PSEN2'], option=['spin'])

    res.plot()

    print('Hi')


def print_hi_multi():
    f = gm.load_example_coregistered()
    res_Y = gm.group_regions(f)
    ctx_l = np.where(np.char.find(res_Y['regionDescriptions'], 'ctx-lh-') != -1)

    img_data = res_Y['data'][ctx_l]
    res, res2 = gm.association(img_data, ['APOE', 'APP', 'PSEN2'], option=['brain', 'spin'])

    res.plot()
    res2.plot()

    print('Hi')


def print_hi2():
    goi = gm.load_example_goi()
    res = gm.geneset_analysis(goi, option=['brain'])

    res.plotBrain()

    print('Hi')


def print_hi3():
    a = gm.PermResImg()
    a.lr.beta = np.load("./lr_beta.npy")
    a.lr.p = np.load("./lr_p.npy")
    a.gene_symbols = np.load("./gene_symbols.npy")

    a.table()


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi()
    print('Hi')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
