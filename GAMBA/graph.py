import atexit
import numbers
import os
import sys
import tempfile
import webbrowser
from typing import Union, Optional
from pathlib import Path

import xml.etree.ElementTree as etree
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Progress:
    def __init__(self, total: int):
        self.total = total
        self.n = 0

    def progress(self, step: int = 1, s: str = '', default=True, percent=False):
        self.n += step
        b = int(self.n / self.total * 30)
        def_str = f'{self.n}/{self.total}' if default else ''
        per_str = ' {:.1f}%'.format(self.n / self.total * 100) if percent else ''
        sys.stdout.write(f'\r## Progress {def_str}:{s} '
                         + '#' * b + '-' * (30 - b)
                         + '|' + per_str)

    def clear(self):
        sys.stdout.write('\r')
        self.n = 0


class Plot:
    fig: plt.Figure
    axs: Union[plt.Axes, np.ndarray[plt.Axes]]

    def __init__(self, nrows=1, ncols=1):
        self.num = nrows * ncols
        if self.num <= 0:
            raise ValueError("nrows * ncols <= 0")
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

    def get_ax(self, nax=1):
        if nax > self.num or nax < 0:
            raise ValueError("nax not in axs region")
        if self.num == 1:
            return self.axs
        else:
            return self.axs[nax - 1]

    def distplot(self, data: np.ndarray, beta: float, p: float, nax=1):
        ax = self.get_ax(nax)
        sns.histplot(data=data, bins=25, kde=True, ax=ax, line_kws={'label': f'p:{p:.3f}'})
        ax.axvline(x=beta, color='r', label='{:.3f}'.format(beta))
        ax.legend()

    def regplot(self, x: np.ndarray, y: np.ndarray, beta: float, p: float, nax=1):
        ax = self.get_ax(nax)
        sns.regplot(x=x, y=y, ax=ax, line_kws={'label': 'beta:{:.3f} p:{:.3f}...'.format(beta, p)})
        ax.legend()

    def show(self):
        self.fig.show()


_atlases = list(['lausanne120', 'aparc', 'aparc_aseg', 'lausanne120_aseg', 'lausanne250', 'wbb47'])
_dirPath = Path(__file__).parent


def plotBrain(regions: np.ndarray[str], values: np.ndarray[numbers.Number], viewer=True,
              limits: Optional[tuple[2]] = None, color: str = 'Blues',
              atlas: str = 'lausanne120', file: Optional[str] = None, scaling=0.1, lh_only=True):
    """
    :param regions: region description
    :param values: value correspond to region
    :param viewer: defined whether the created figure will be opened in web viewer
    :param limits: Two elements tuple [cmin, cmax] limits display range
    :param color: Name of matplotlib colormap
    :param file: path to save the file
    :param scaling: scale of the figure, 0.1 by default
    :param lh_only: show left hemisphere only
    :param atlas: 'lausanne120'(default), 'aparc', 'aparc_aseg', 'lausanne120_aseg', 'lausanne250', 'wbb47'

    """
    regions = regions.flatten()
    values = values.flatten()

    if regions.size != values.size:
        raise ValueError("regions' size is not the same values")

    if atlas not in _atlases:
        raise ValueError("atlas", atlas, "is not supported")

    if limits is None:
        limits = [np.min(values), np.max(values)]

    values[values < limits[0]] = limits[0]
    values[values > limits[1]] = limits[1]

    colors = np.round(np.array(sns.color_palette(color)) * 255).astype(int)
    values_norm = (values - limits[0]) / (limits[1] - limits[0])
    values_color_idx = (np.round(values_norm * (colors.shape[0] - 1))).astype(int)

    # savefile and cb_file
    if file is None:
        tmpFp = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        savefile = tmpFp.name
        savefile = Path(savefile).with_suffix('.svg').__str__()
        tmpFp.close()
    else:
        savefile = Path(file).resolve().__str__()  # absolute
    cb_file = savefile + '_cb.png'

    # color bar
    _RGB_to_File(colors, cb_file)

    # overwrite svg
    # load svg
    template = 'lh_' + atlas + '_template.svg' if (atlas != 'wbb47' and lh_only) else atlas + '_template.svg'
    template_path = _dirPath.joinpath('default', 'graph', template).__str__()
    tree = etree.ElementTree(file=template_path)
    root = tree.getroot()
    elems = root.iter('{http://www.w3.org/2000/svg}path')

    # scaling
    root.set('height', '{}mm'.format(1756 * scaling))
    root.set('width', '{}mm'.format(4224.5 * scaling))

    # coloring
    for elem in elems:
        id = elem.get('id')
        if id is None or 'ctx-' not in id:
            continue
        # print(id)
        if 'pericalcarine' in id:
            a = 1
        for rk, r in enumerate(regions):
            if r in id:
                cid = values_color_idx[rk]
                elem.set('fill', _RGB_to_Hex(colors[cid]))
                if lh_only:
                    elem.set('data', str(values[rk]))
                    elem.set('onmouseover', 'onhover(this)')

    # set label
    label = root.find('{http://www.w3.org/2000/svg}g[@id="g10000"]')
    label[0].set('{http://www.w3.org/1999/xlink}href', Path(cb_file).name)
    label[1][0].text = '{:.4f}'.format(limits[1])
    label[2][0].text = '{:.4f}'.format(limits[0])

    # save svg
    tree.write(savefile)
    # show svg
    if viewer:
        webbrowser.open_new_tab(os.path.abspath(savefile))

    if file is None:
        atexit.register(_clean_file, savefile, cb_file)


def _clean_file(f1, f2):
    os.remove(f1)
    os.remove(f2)


def _RGB_to_Hex(rgb):
    strs = '#'
    for i in rgb:
        # num = int(i * 255)  # 将str转int
        # 将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(i))[-2:].replace('x', '0').upper()

    return strs


def _RGB_to_File(rgb: np.ndarray, file):
    lines = rgb.shape[0]
    im = PIL.Image.new("RGB", (1, lines))
    for i in range(lines):
        # im.putpixel((0, lines - 1 - i), (int(np.round(rgb[i][0] * 255)), int(np.round(rgb[i][1] * 255)),
        #                                  int(np.round(rgb[i][2] * 255)))))
        im.putpixel((0, lines - 1 - i), tuple(rgb[i].tolist()))
    im.save(file)
