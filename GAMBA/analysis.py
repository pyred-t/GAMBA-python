from typing import List

import numpy as np

from .null import *
from .null_exp import *

__opts = dict({
    'spin': [permutation_null_spin, permutation_expression_null_spin],
    'coexp': [permutation_null_coexp, permutation_expression_null_coexp],
    'brain': [permutation_null_brain, permutation_expression_null_brain]
})


def association(img_data: np.ndarray, goi: List[str], option: Union[str, List[str]], gene: Gene = Gene(),
                background: Optional[str] = None) -> Union[PermRes, List[PermRes]]:
    if isinstance(option, str):
        option = list(option)

    if not (0 < option.__len__() < 3):
        raise ValueError("too many or too few options")

    for opt in option:
        if opt not in __opts:
            raise ValueError("option error")

    res = list()
    goi = np.array(goi)

    for opt in option:
        if opt == 'brain':
            background = 'brain' if not background else background
            res.append(__opts[opt][0](img_data, goi, gene.expression, gene.symbols, background))
        else:
            res.append(__opts[opt][0](img_data, goi, gene.expression, gene.symbols))

    if res.__len__() == 1:
        return res[0]
    else:
        return res


def geneset_analysis(goi: List[str], option: Union[str, List[str]], gene: Gene = Gene(),
                     background: Optional[str] = None) -> Union[PermResExp, List[PermResExp]]:
    if isinstance(option, str):
        option = list(option)

    if not (0 < option.__len__() < 3):
        raise ValueError("too many or too few options")

    for opt in option:
        if opt not in __opts:
            raise ValueError("option error")

    res = list()
    goi = np.array(goi)

    for opt in option:
        if opt == 'brain':
            background = 'brain' if not background else background
            res.append(__opts[opt][1](goi, gene.expression, gene.symbols, background))
        else:
            res.append(__opts[opt][1](goi, gene.expression, gene.symbols))

    if res.__len__() == 1:
        return res[0]
    else:
        return res


def image_based_analysis(img_data: np.ndarray) -> PermResImg:
    return permutation_null_spin_corr(img_data)
