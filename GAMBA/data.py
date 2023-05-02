import os
import random
import sys
import tempfile
# import urllib.request
import time
import zipfile
from functools import partial
from typing import Optional
import platform
import requests

import nibabel
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat

from .graph import Progress

import multiprocessing.dummy as mp

dirPath = Path(__file__).parent


class Gene:
    expression: np.ndarray
    symbols: np.ndarray
    regionDescriptions: np.ndarray

    def __init__(self, exp=None, symbols=None, regionDesc=None):
        self.expression = exp
        self.symbols = symbols
        self.regionDescriptions = regionDesc


def load_gene_expression(regionDesc=False, expression=True, gene_symbols=True) -> Gene or None:
    if not (expression or gene_symbols or regionDesc):
        return None
    ge_path = dirPath.joinpath('default', 'gene_expression.mat')
    data = loadmat(ge_path.__str__(), simplify_cells=True)
    gene = Gene()
    gene.expression = data['mDataGEctx'] if expression else None
    gene.symbols = data['gene_symbols'] if gene_symbols else None
    gene.regionDescriptions = data['regionDescriptionCtx'] if regionDesc else None
    return gene


def fetch_null_spin():
    print('load default spin data(~1.5G), make sure the connection to Dropbox')
    url = 'https://www.dropbox.com/s/nwmtqro3dkma3u1/gene_expression_spin.zip?dl=1'
    # url = 'https://189.ly93.cc/NBr6Zb6RrEBz/12529163552364716?accessCode=sa67'

    temp = tempfile.NamedTemporaryFile(mode='wb', delete=False)
    # print(temp.name)
    # temp.close()
    # urllib.request.urlretrieve(url, filename=temp.name)

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        raise ConnectionError("Unable to connect to the files")
    chunk_size = 1024
    content_size = int(response.headers['content-length'])
    pbar = Progress(content_size)

    chunk_count = 0
    chunk_thread = 1024*100
    speed_str = ''
    t = time.time()

    for data in response.iter_content(chunk_size=chunk_size):
        temp.write(data)
        chunk_count += len(data)
        if chunk_count > chunk_thread:
            speed = (chunk_count / 1024) / (time.time() - t)
            t = time.time()
            chunk_count = 0
            speed_str = '{:.2f}MB/s'.format(speed / 1024) if speed > 1024 else '{:.2f}kB/s'.format(speed)

        pbar.progress(len(data), s=speed_str, default=False, percent=True)

    pbar.clear()
    temp.close()

    print("download ok, retrieving...")

    spinDir = dirPath.joinpath('default', 'gene_expression_spin')
    if not spinDir.is_dir():
        os.mkdir(spinDir.__str__())

    zfile = zipfile.ZipFile(temp.name)
    zfile.extractall(spinDir.__str__())
    zfile.close()

    os.remove(temp.name)
    print('>> ok')


def load_null_spin(gene: str) -> np.ndarray:
    spinDir = dirPath.joinpath('default', 'gene_expression_spin')
    if not spinDir.is_dir():
        raise FileNotFoundError('Spin data not found. To use null-spin-model, please use fetch_null_spin() to fetch '
                                'the spin data first')

    spinFile = spinDir.joinpath('GE_spin_' + gene + '.txt')
    if spinFile.is_file():
        return np.loadtxt(spinFile.__str__(), delimiter=',', ndmin=1)
    else:
        Warning("spin data not found, return empty array")
        return np.array([])


_bg_idx = dict({
    "brain": "BRAINgene_idx",
    "body": "BRAINandBODYgene_idx",
    "general": "BRAIN_expressed_gene_idx"
})


def load_gene_expression_background(background: str) -> Gene:
    """
    :param background: "brain", "body", "general"
    """
    if background not in _bg_idx.keys():
        Warning("Background genes are not properly selected. Setting to 'brain' by default.")
        background = "brain"
    ge_path = dirPath.joinpath('default', 'gene_expression.mat')
    data = loadmat(ge_path.__str__(), simplify_cells=True)
    gene = Gene()
    gene.symbols = data['gene_symbols'][data[_bg_idx[background]] - 1]
    return gene


def coregister(output_reg_mat: str, output_img_file: str,
               img_file: Optional[str] = None, img_anat_file: Optional[str] = None, ref_img_file: Optional[str] = None):
    """
    let brain map (.nii file) be co-registered to the same space as MNI152 brain, i.e.

    """
    if platform.system() != 'Linux' or 'version' not in os.system('flirt -version'):
        raise ValueError("Supported Linux with flirt only. Try to use other tools to replace this function, "
                         "or use load_example_coregistered() to load the default example")

    egDir = dirPath.joinpath('default', 'examples')
    if not img_file:
        img_file = egDir.joinpath('alzheimers_ALE.nii.gz').resolve().__str__()
    if not img_anat_file:
        img_anat_file = egDir.joinpath('Colin27_T1_seg_MNI_2x2x2.nii.gz').resolve().__str__()
    if not ref_img_file:
        ref_img_file = egDir.joinpath('brain.nii.gz').resolve().__str__()

    os.system('flirt -in ' + img_anat_file + ' -ref ' + ref_img_file + ' -omat ' + output_reg_mat)
    os.system('flirt -in ' + img_file + ' -ref ' + ref_img_file
              + '-applyxfm -init' + output_reg_mat + '-out' + output_img_file)

    print("coregister ok")


def load_example_coregistered() -> str:
    """
    load alzheimer's VBM img that has been coregistered to MNI152 brain
    :return: the coregistered file path
    """
    egDir = dirPath.joinpath('default', 'examples', 'out')
    return egDir.joinpath('coreg_alzheimers_ALE.nii.gz').resolve().__str__()


def load_example_goi():
    ge_path = dirPath.joinpath('default', 'examples', 'example_conn_5k_genes.mat')
    data = loadmat(ge_path.__str__(), simplify_cells=True)
    return data['geneset']


_atlases = dict({
    'DK114': ['lausanne120.txt', 'lausanne120+aseg.nii.gz'],
    'aparc': ['aparc.txt', 'aparc+aseg.nii.gz'],
    'DK250': ['lausanne250.txt', 'lausanne250+aseg.nii.gz']
})


def group_regions(co_img_file: str, atlas: str = 'DK114') -> dict[str, np.ndarray]:
    """
    :param co_img_file: input brain map (.nii file) that has been co-registered to the same space as MNI152 brain, i.e.
    :param atlas: 'DK114'(default), 'aparc', 'DK250'
    :return:
        data -- N x 1 array of regional mean value extracted from the input imaging data. N is the number of regions.
        regionDescriptions -- N  region descriptions.
        regionIndexes -- N  region indexes.
    """
    if atlas not in _atlases.keys():
        raise ValueError("Atlas", atlas, "is not supported.")
    atl_path = dirPath.joinpath('default', 'atlas')
    lookupTable = atl_path.joinpath(_atlases[atlas][0]).__str__()
    ref_file = atl_path.joinpath(_atlases[atlas][1]).__str__()

    print(f'group regions by atlas {atlas}')

    hdr = _load_nifti(co_img_file)
    vol = hdr['vol']

    ref = _load_nifti(ref_file)

    res = dict()
    # read color table
    tbl = pd.read_csv(lookupTable, sep="\\s+", header=None)
    res['regionIndexes'] = tbl.iloc[:, 0].values
    res['regionDescriptions'] = tbl.iloc[:, 1].values.astype(str)

    # compute regional mean
    # res['data'] = np.full((res['regionDescriptions'].size, 1), np.nan)

    # t = time.time()
    # tbl['a'] = tbl.iloc[:, 0].apply(_myfunc, ref_vol=ref['vol'], vol=vol)
    pfunc = partial(_myfunc, ref_vol=ref['vol'], vol=vol)
    with mp.Pool() as tpool:
        res['data'] = np.array(tpool.map(pfunc, res['regionIndexes']))
    # print(f'1 {time.time() - t}')

    # t = time.time()
    # for i, x in enumerate(res['regionIndexes']):
    #     tmp = vol[ref['vol'] == x]
    #     res['data'][i] = np.mean(tmp)
    # print(f'1 {time.time() - t}')

    return res

def _myfunc(x, ref_vol, vol):
    return np.mean(vol[ref_vol == x])


def _load_nifti(nifti_file) -> dict:
    res = dict()
    hdr = nibabel.load(nifti_file)

    res['header'] = hdr.header
    res['vol'] = hdr.get_fdata()

    return res
