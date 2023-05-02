import scipy

from .data import *
from .graph import *

from typing import Optional

import numpy as np


class Lrg:
    beta: np.ndarray
    p: np.ndarray
    X: np.ndarray  # gene_expression
    Y: np.ndarray  # img_data


class PermRes:
    p: np.ndarray
    lr: Lrg
    permut_beta: np.ndarray
    permut_gene_idx: np.ndarray
    coexp_mean: np.ndarray
    permut_coexp_mean: np.ndarray

    graph: Plot

    def __init__(self):
        self.lr = Lrg()

    def plot(self, reg=True, dist=True, view=True, save_fig: Optional[str] = None):
        if not (reg or dist):
            return
        # if nrows * ncols < reg + dist:
        #     raise ValueError("could not load enough figure subplots")
        if (not hasattr(self, 'graph')) or self.graph.num < reg + dist:
            if reg and dist:
                self.graph = Plot(1, 2)
            else:
                self.graph = Plot(1, 1)

        nax = 1
        if reg:
            if (not hasattr(self, 'lr')) or hasattr(self, 'gene_symbols'):
                raise ValueError("logical regression data not found")
            M = self.lr.Y.shape[1]
            for i in range(M):
                self.graph.regplot(self.lr.X, self.lr.Y[:, i], self.lr.beta[i, 0], self.lr.p[i, 0], nax=nax)

            self.graph.get_ax(nax).set_xlabel('gene_expression')
            self.graph.get_ax(nax).set_ylabel('img_data')
            self.graph.get_ax(nax).set_title('image data and gene expression scatter and linear plot')

            nax += 1
        if dist:
            if not (hasattr(self, 'permut_beta') and hasattr(self, 'lr')):
                raise ValueError("permutation data or logical regression data not found")
            M = self.permut_beta.shape[1]
            for j in range(M):
                self.graph.distplot(self.permut_beta[:, j], self.lr.beta[j, 0], nax=nax)

            self.graph.get_ax(nax).set_xlabel('permutation_beta')
            self.graph.get_ax(nax).set_title('null distribution plot')

        if view:
            self.graph.show()
        if save_fig:
            self.graph.fig.savefig(save_fig)


class PermResImg:
    p: np.ndarray
    lr: Lrg
    gene_symbols: np.ndarray

    graph: Plot

    def __init__(self):
        self.lr = Lrg()

    def table(self, TopK=10, p=0.5, view=True, save_fig: Optional[str] = None):
        if not (hasattr(self, 'lr') and hasattr(self, 'gene_symbols')):
            raise ValueError("table data not found")

        M = self.lr.beta.shape[1]
        self.graph = Plot(M)
        for i in range(M):
            beta_sort = np.argsort(self.lr.beta[:, i])
            m_invalid = ~np.logical_or(np.isnan(self.lr.beta[:, i]), self.lr.p[:, i] > p)
            m_invalid = m_invalid[beta_sort]
            beta_sort = beta_sort[m_invalid]

            TopK = TopK if beta_sort.size >= TopK else beta_sort.size
            max_beta_id = beta_sort[-TopK:][::-1]
            info = np.empty([TopK, 2])
            info[:, 0] = np.around(self.lr.beta[:, i][max_beta_id],4)
            info[:, 1] = np.around(self.lr.p[:, i][max_beta_id], 3)

            ax = self.graph.get_ax(i + 1)
            tab = ax.table(cellText=info, loc='center', rowLabels=self.gene_symbols[max_beta_id],
                           colLabels=['beta', 'p'], cellLoc='center', colWidths=[0.3, 0.2])
            tab.auto_set_font_size(True)
            tab.scale(1.2,1.2)
            ax.axis('off')
            ax.set_title(f'Top {TopK} most relational genes expression')

        if view:
            self.graph.fig.show()
        if save_fig:
            self.graph.fig.savefig(save_fig)


def permutation_null_spin(img_data: np.ndarray, goi: np.ndarray,
                          expressions: Optional[np.ndarray] = None, gene_symbols: Optional[np.ndarray] = None):
    """

    :param img_data: NxM,N:brain regions, M: imaging traits
    :param goi: 1xN gene symbols of the genes of interest
    :param expressions: NxK, N:number of regions, K:number of genes
    :param gene_symbols: 1xN gene symbols
    :return:
    """

    # ========== load data ==========
    print('Running null-spin model')
    if expressions is None and gene_symbols is None:
        gene = load_gene_expression(regionDesc=False)
    elif expressions is None or gene_symbols is None:
        raise ValueError("'Please provide gene symbols of all genes included in the expression data")
    else:
        gene = Gene(expressions, gene_symbols)

    mask = _check_img_geneExp(img_data, goi, gene)

    N, M = img_data.shape
    # ========== perform linear regression ==========
    res = PermRes()
    beta = np.full((M, 1), np.nan)
    pval = np.full((M, 1), np.nan)

    # standardize
    ETmp = (gene.expression[np.tile(mask.T, (N, 1))]).reshape((N, mask.sum()))
    X = np.nanmean(ETmp, 1)
    X = (X - np.nanmean(X)) / np.nanstd(X)
    Y = img_data
    Y = (Y - np.full((N, 1), np.nanmean(Y, 0))) / np.full((N, 1), np.nanstd(Y, 0))

    for i in range(M):
        stats = scipy.stats.linregress(Y[:, i], X)
        beta[i, 0] = stats.slope
        pval[i, 0] = stats.pvalue

    res.lr.beta = beta
    res.lr.p = pval
    res.lr.X = X
    res.lr.Y = Y

    # ========== perform permutation ==========
    # TODO: config perm times setting
    NPerm = 1000
    null_spin_expression = np.full([NPerm, 57, goi.size], np.nan)
    for i, v in enumerate(goi):
        null_spin_expression[:, :, i] = load_null_spin(v)

    null_spin_exp_mean = np.nanmean(null_spin_expression, 2)

    pbar = Progress(NPerm)
    beta_null = np.full((NPerm, M), np.nan)

    for k in range(NPerm):
        pbar.progress()
        # randomized gene expressions
        X = null_spin_exp_mean[k, :]
        X = (X - np.nanmean(X)) / np.nanstd(X)

        for i in range(M):
            # pair-wise, ignoring nan row
            mask_invalid = np.logical_or(np.isnan(Y[:, i]), np.isnan(X))
            stats = scipy.stats.linregress(Y[:, i][~mask_invalid], X[~mask_invalid])
            beta_null[k, i] = stats.slope

    pbar.clear()

    res.permut_beta = beta_null

    # ===== compute p-value =====
    res.p = np.full((M, 1), np.nan)
    for i in range(M):
        P = np.count_nonzero((beta_null[:, i] > beta[i])) / NPerm
        if P > 0.5:
            res.p[i, 0] = (1 - P) * 2
        else:
            res.p[i, 0] = P * 2

    print(" >> finished without errors")
    return res


def permutation_null_brain(img_data: np.ndarray, goi: np.ndarray, expressions: Optional[np.ndarray] = None,
                           gene_symbols: Optional[np.ndarray] = None, background: str = 'brain'):
    """

    :param img_data: NxM,N:brain regions, M: imaging traits
    :param goi: 1xN gene symbols of the genes of interest
    :param expressions: NxK, N:number of regions, K:number of genes
    :param gene_symbols: 1xN gene symbols
    :param background: "brain"--default, "body", "general"
    """
    print('Running null-brain model')

    if expressions is None and gene_symbols is None:
        gene = load_gene_expression(regionDesc=False)
    elif expressions is None or gene_symbols is None or expressions.size == 0 or gene_symbols.size == 0:
        raise ValueError("'Please provide gene symbols of all genes included in the expression data")
    else:
        gene = Gene(expressions, gene_symbols)

    _check_img_geneExp(img_data, goi, gene)

    ref_ge = load_gene_expression_background(background)

    goi = np.intersect1d(goi, ref_ge.symbols)
    mask = np.isin(gene.symbols, goi)
    print('##', mask.sum(), 'genes of the input GOI are background-enriched genes.')

    N, M = img_data.shape
    # ========== perform linear regression ==========
    res = PermRes()
    beta = np.full((M, 1), np.nan)
    pval = np.full((M, 1), np.nan)

    # standardize
    ETmp = (gene.expression[np.tile(mask.T, (N, 1))]).reshape((N, mask.sum()))
    X = np.nanmean(ETmp, 1)
    X = (X - np.nanmean(X)) / np.nanstd(X)
    Y = img_data
    Y = (Y - np.full((N, 1), np.nanmean(Y, 0))) / np.full((N, 1), np.nanstd(Y, 0))

    for i in range(M):
        stats = scipy.stats.linregress(Y[:, i], X)
        beta[i, 0] = stats.slope
        pval[i, 0] = stats.pvalue

    res.lr.beta = beta
    res.lr.p = pval
    res.lr.X = X
    res.lr.Y = Y

    # ========== perform permutation ==========
    # TODO:config
    NPerm = 10000

    idx_rand_genes = np.full((NPerm, mask.sum()), np.nan)
    beta_null = np.full((NPerm, M), np.nan)

    idx_background = _iloc_isMember(ref_ge.symbols, gene.symbols)

    pbar = Progress(NPerm)
    # permutation
    for k in range(NPerm):
        pbar.progress()
        rid: np.ndarray = idx_background[np.random.permutation(idx_background.size)[:mask.sum()]]
        idx_rand_genes[k, :] = rid

        # gene expressions of random genes
        X = np.nanmean(gene.expression[:, rid], 1)
        X = (X - np.nanmean(X)) / np.nanstd(X)

        for i in range(M):
            stats = scipy.stats.linregress(Y[:, i], X)
            beta_null[k, i] = stats.slope

    pbar.clear()

    res.permut_gene_idx = idx_rand_genes
    res.permut_beta = beta_null

    # ===== compute p-value =====
    res.p = np.full((M, 1), np.nan)
    for i in range(M):
        P = np.count_nonzero((beta_null[:, i] > beta[i])) / NPerm
        if P > 0.5:
            res.p[i, 0] = (1 - P) * 2
        else:
            res.p[i, 0] = P * 2

    print(" >> finished without errors")
    return res


def permutation_null_coexp(img_data: np.ndarray, goi: np.ndarray,
                           expressions: Optional[np.ndarray] = None, gene_symbols: Optional[np.ndarray] = None):
    """
    :param img_data: NxM,N:brain regions, M: imaging traits
    :param goi: 1xN gene symbols of the genes of interest
    :param expressions: NxK, N:number of regions, K:number of genes
    :param gene_symbols: 1xN gene symbols
    :return:
    """
    print('Running null-coexp model')
    if expressions is None and gene_symbols is None:
        gene = load_gene_expression(regionDesc=False)
    elif expressions is None or gene_symbols is None:
        raise ValueError("'Please provide gene symbols of all genes included in the expression data")
    else:
        gene = Gene(expressions, gene_symbols)

    mask = _check_img_geneExp(img_data, goi, gene)

    N, M = img_data.shape
    # ========== perform linear regression ==========
    res = PermRes()
    beta = np.full((M, 1), np.nan)
    pval = np.full((M, 1), np.nan)

    # standardize
    ETmp = (gene.expression[np.tile(mask.T, (N, 1))]).reshape((N, mask.sum()))
    X = np.nanmean(ETmp, 1)
    X = (X - np.nanmean(X)) / np.nanstd(X)
    Y = img_data
    Y = (Y - np.full((N, 1), np.nanmean(Y, 0))) / np.full((N, 1), np.nanstd(Y, 0))

    for i in range(M):
        stats = scipy.stats.linregress(Y[:, i], X)
        beta[i, 0] = stats.slope
        pval[i, 0] = stats.pvalue

    res.lr.beta = beta
    res.lr.p = pval
    res.lr.X = X
    res.lr.Y = Y

    # ========== perform permutation ==========
    NPerm = 1000

    # compute coexpression of the input GOI
    G = gene.expression[np.tile(mask.T, (N, 1))]
    G = G.reshape((N, mask.sum()))
    coexp_mat = np.ma.corrcoef(np.ma.masked_invalid(G), rowvar=False).data
    mask_tri = np.tril(np.ones(coexp_mat.shape), -1)
    coexp = np.nanmean(coexp_mat[mask_tri == 1])

    res.coexp_mean = coexp

    coexp_null = np.full((NPerm, 1), np.nan)
    idx_rand_genes = np.full((NPerm, mask.sum()), np.nan)
    beta_null = np.full((NPerm, M), np.nan)

    pbar = Progress(NPerm)

    for k in range(NPerm):
        tmp_status = False
        rid = None
        while tmp_status is not True:
            rid, coexp_null[k], tmp_status = _y_rand_gs_coexp(gene.expression, coexp, np.count_nonzero(mask))
        pbar.progress()

        idx_rand_genes[k, :] = rid

        # gene expressions of random genes
        X = np.nanmean(gene.expression[:, rid], 1)
        X = (X - np.nanmean(X)) / np.nanstd(X)

        for i in range(M):
            stats = scipy.stats.linregress(Y[:, i], X)
            beta_null[k, i] = stats.slope

    pbar.clear()

    res.permut_gene_idx = idx_rand_genes
    res.permut_beta = beta_null
    res.permut_coexp_mean = coexp_null

    # ===== compute p-value =====
    res.p = np.full((M, 1), np.nan)
    for i in range(M):
        P = np.count_nonzero((beta_null[:, i] > beta[i])) / NPerm
        if P > 0.5:
            res.p[i, 0] = (1 - P) * 2
        else:
            res.p[i, 0] = P * 2

    print(" >> finished without errors")
    return res


def permutation_null_spin_coexp(img_data):
    print('loop all genes, looking for top correlated genes for img_data')

    gene = load_gene_expression(regionDesc=False)

    N, M = img_data.shape
    print("##", N, "brain regions detected,", M, "imaging traits detected.")

    NE, K = gene.expression.shape
    if N != NE:
        raise ValueError("Different amount of regions in imaging data and gene data.")
    print("##", K, "genes detected totally")

    # =================== loop all genes ===================
    res = PermResImg()
    # K:gene_num M:img_data_group
    beta = np.full((K, M), np.nan)
    pval = np.full((K, M), np.nan)

    res.p = np.full((K, M), np.nan)

    # standardize
    Y = img_data
    Y = (Y - np.full((N, 1), np.nanmean(Y, 0))) / np.full((N, 1), np.nanstd(Y, 0))

    pbar = Progress(K)
    for i in range(K):
        pbar.progress()

        X = gene.expression[:, i]
        X = (X - np.nanmean(X)) / np.nanstd(X)

        beta_null = np.full((1000, M), np.nan)

        for j in range(M):
            # linear regression
            stats = scipy.stats.linregress(Y[:, j], X)
            beta[i, j] = stats.slope
            pval[i, j] = stats.pvalue
            # null_spin
            null_spin_expression = load_null_spin(gene.symbols[i]).T
            if null_spin_expression.size == 0:
                null_spin_expression = np.full((N, 1000), np.nan)

            for k in range(1000):
                X = null_spin_expression[:, k]
                if not np.isnan(X).all():
                    X = (X - np.nanmean(X)) / np.nanstd(X)
                    # linear regression
                    stats = scipy.stats.linregress(Y[:, j], X)
                    beta_null[k, j] = stats.pvalue
            # pvalue
            P = np.count_nonzero((beta_null[:, j] > beta[i, j])) / 1000
            if P > 0.5:
                res.p[i, j] = (1 - P) * 2
            else:
                res.p[i, j] = P * 2
        # end for in loop M
    # end for in loop K
    pbar.clear()

    res.lr.beta = beta
    res.lr.p = pval
    res.gene_symbols = gene.symbols

    return res


def _check_img_geneExp(img_data: np.ndarray, goi: np.ndarray, gene: Gene) -> np.ndarray:
    """
    :return mask on gene.symbols with goi
    """
    N, M = img_data.shape
    print("##", N, "brain regions detected,", M, "imaging traits detected.")

    NE, K = gene.expression.shape
    if N != NE:
        raise ValueError("Different amount of regions in imaging data and gene data.")
    print("##", K, "genes detected totally")

    NG = goi.size
    print("##", NG, "genes of the GOI detected.")
    if gene.symbols.size != K:
        raise ValueError("The number of gene symbols is different from the number of genes in the expression data")

    mask = np.isin(gene.symbols, goi)
    nnz_mask = np.count_nonzero(mask)
    if nnz_mask == 0:
        raise ValueError("None of the genes in the input gene set found in gene data")
    print("##", nnz_mask, '/', NG, 'genes with gene expression data available')

    return mask


def _iloc_isMember(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A在B中出现元素的第一个位置，不包含不出现元素
    """
    _, unqx = np.unique(B, return_index=True)
    inx = np.where(np.isin(B, A))

    return np.intersect1d(unqx, inx, assume_unique=True)


def _y_rand_gs_coexp(G: np.ndarray, T, NG: int, maxDiff=0.025):
    NGenes = G.shape[1]
    NCount = 0
    status = True

    # initialize a set of random genes
    gene_id: np.ndarray = np.random.permutation(NGenes)[:NG]
    GRand = G[:, gene_id]
    # GRand = G[:, :100]

    # compute mean coexpression for each gene and for the whole set
    _, tmp_coexp_mean, coexp = _compute_mean_coexp(GRand)
    delta_coexp = coexp - T  # difference from the Target

    # compute coexpression between the GOI and the rest of genes
    GSub_G_coexp = _compute_GSub_G_coexp(GRand, G)

    GSub_G_coexp_sorted_idx = np.argsort(GSub_G_coexp).astype(int)
    GSub_G_coexp_sorted_idx = GSub_G_coexp_sorted_idx[~np.isin(GSub_G_coexp_sorted_idx, gene_id)]

    NRep = int(np.ceil(NG / 100))  # the number of genes to be replaced during iterations

    while abs(delta_coexp) > maxDiff:
        NCount += 1

        # need to decrease coexpression level
        if delta_coexp > 0:
            # find the gene whit max coexpression
            idx_set = np.argpartition(tmp_coexp_mean, -NRep)[-NRep:]

            # add new genes
            rid = GSub_G_coexp_sorted_idx[:NRep]  # replace with the one differs the most
            GSub_G_coexp_sorted_idx = np.delete(GSub_G_coexp_sorted_idx, (0, NRep))

        # increase coexpression level
        else:
            # find the gene whit min coexpression
            idx_set = np.argpartition(tmp_coexp_mean, NRep)[:NRep]

            # add new genes
            rid = GSub_G_coexp_sorted_idx[-NRep:]  # replace with the one differs the most
            GSub_G_coexp_sorted_idx = np.delete(GSub_G_coexp_sorted_idx, (-NRep,))
        # end if

        # replace
        GRand[:, idx_set] = G[:, rid]
        gene_id[idx_set] = rid
        # compute delta coexp again
        _, tmp_coexp_mean, coexp = _compute_mean_coexp(GRand)
        delta_coexp = coexp - T

        # Finished: 500
        if NCount == 300:
            status = False
            break
    # end while
    return gene_id, coexp, status


# compute mean coexpression
def _compute_mean_coexp(G_sub):
    coexp_mat = _corr2_coeff(G_sub, G_sub, rowvar=False, nan='pairwise')
    coexp_mean = np.nanmean(coexp_mat, 0)
    mask_tril = np.tril(np.ones(coexp_mean.shape), -1)
    coexp_mean_mean = np.nanmean(coexp_mat[mask_tril == 1])
    return coexp_mat, coexp_mean, coexp_mean_mean


# compute coexpression from geneset to all
def _compute_GSub_G_coexp(GSub, G):
    rtmp = _corr2_coeff(GSub, G, rowvar=False, nan='pairwise')
    GSub_G_coexp = np.nanmean(rtmp, 0)
    return GSub_G_coexp


def _corr2_coeff(A, B, rowvar: Optional[bool] = True, nan: Optional[str] = 'all'):
    if rowvar is False:
        A = A.T
        B = B.T

    # Row-wise mean of input arrays & subtract from input arrays themselves
    # cov(X,Y) = E[(X-EX)*(Y-EY)]
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    # DX = sum{(X-EX)^2}/N
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    # Finally get corr coeff
    res = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    if nan == 'pairwise':
        A_null_idx = np.unique(np.where(np.isnan(A))[0])
        B_null_idx = np.unique(np.where(np.isnan(B))[0])

        for i in A_null_idx:
            for j in range(B.shape[0]):
                res[i, j] = _corr1_coeff(A[i], B[j], nan='pairwise')

        for j in B_null_idx:
            for i in range(A.shape[0]):
                res[i, j] = _corr1_coeff(A[i], B[j], nan='pairwise')

    return res


def _corr1_coeff(A, B, nan: Optional[str] = 'all'):
    if nan == 'pairwise':
        idx = np.union1d(np.where(np.isnan(A))[0], np.where(np.isnan(B))[0])
        A = np.delete(A, idx)
        B = np.delete(B, idx)

    A_mA = A - A.mean()
    B_mB = B - B.mean()

    ssA = (A_mA ** 2).sum()
    ssB = (B_mB ** 2).sum()

    res = np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA, ssB))
    return res
