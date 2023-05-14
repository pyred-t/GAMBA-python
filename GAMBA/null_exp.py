from .data import *
from .graph import Progress, plotBrain
from .null import _iloc_isMember, _y_rand_gs_coexp


class PermResExp:
    p: np.ndarray
    mean_expression: np.ndarray
    null_expression: np.ndarray
    difference: np.ndarray
    regionDescriptions: np.ndarray
    coexp_mean: np.ndarray
    permut_gene_idx: np.ndarray
    permut_coexp_mean: np.ndarray
    null_expressions: np.ndarray
    difference: np.ndarray

    def plotBrain(self, viewer=True, limits: Optional[tuple[2]] = None, color: str = 'Blues',
                  atlas: str = 'lausanne120', file: Optional[str] = None, scaling=0.1, lh_only=True):
        """
        draw the expression level on brain base on the atlas selected, according to regionDescriptions and mean_expression.

        :param viewer: defined whether the created figure will be opened in web viewer
        :param limits: Two elements tuple [cmin, cmax] limits display range
        :param color: Name of matplotlib colormap
        :param file: path to save the file
        :param scaling: scale of the figure, 0.1 by default
        :param lh_only: show left hemisphere only
        :param atlas: 'lausanne120'(default), 'aparc', 'aparc_aseg', 'lausanne120_aseg', 'lausanne250', 'wbb47'
        """
        if not (hasattr(self, 'regionDescriptions') and hasattr(self, 'mean_expression')):
            raise ValueError('regionDescriptions and mean_expression are not found')

        plotBrain(self.regionDescriptions, self.mean_expression, viewer, limits,
                  color, atlas, file, scaling, lh_only)


def permutation_expression_null_spin(goi, expressions: Optional[np.ndarray] = None,
                                     gene_symbols: Optional[np.ndarray] = None,
                                     regionDesc: Optional[str] = None) -> PermResExp:
    print("Running null-spin model")

    if expressions is None and gene_symbols is None:
        gene = load_gene_expression(True)
    elif expressions is None or gene_symbols is None:
        raise ValueError("Please provide gene symbols of all genes included in the expression data.")
    else:
        gene = Gene(expressions, gene_symbols, regionDesc)

    mask = _check_goi(goi, gene)
    goi: np.ndarray = gene.symbols[mask]

    N, K = gene.expression.shape
    # ========== perform permutation ==========
    res = PermResExp()
    G = gene.expression[np.tile(mask.T, (N, 1))]
    G = G.reshape((N, mask.sum()))
    meanGE = np.nanmean(G, 1)
    res.mean_expression = meanGE

    NPerm = 1000
    # load spin
    null_spin_expression = np.full([NPerm, 57, goi.size], np.nan)
    for i, v in enumerate(goi):
        null_spin_expression[:, :, i] = load_null_spin(v)

    null_spin_exp_mean = np.nanmean(null_spin_expression, 2).T
    res.null_expression = null_spin_exp_mean

    # ===== compute p-value =====
    res.p = np.full((N, 1), np.nan)
    for i in range(N):
        P = np.count_nonzero((null_spin_exp_mean[i, :] > meanGE[i])) / NPerm
        if P > 0.5:
            res.p[i, 0] = (1 - P) * 2
        else:
            res.p[i, 0] = P * 2

    res.difference = res.mean_expression - np.nanmean(res.null_expression, 1)
    res.regionDescriptions = gene.regionDescriptions

    print(" >> finished without errors")
    return res


def permutation_expression_null_brain(goi, expressions: np.ndarray = None,
                                      gene_symbols: np.ndarray = None, regionDesc: Optional[str] = None,
                                      background: str = 'brain') -> PermResExp:
    print('Running null-brain model')
    if expressions is None and gene_symbols is None:
        gene = load_gene_expression(True)
    elif expressions is None or gene_symbols is None:
        raise ValueError("Please provide gene symbols of all genes included in the expression data.")
    else:
        gene = Gene(expressions, gene_symbols, regionDesc)

    _check_goi(goi, gene)

    ref_ge = load_gene_expression_background(background)

    goi = np.intersect1d(goi, ref_ge.symbols)
    mask = np.isin(gene.symbols, goi)
    print('##', mask.sum(), 'genes of the input GOI are background-enriched genes.')

    N, K = gene.expression.shape
    # ========== perform permutation ==========
    res = PermResExp()
    NPerm = 1000

    # row mean gene expression
    G = gene.expression[np.tile(mask.T, (N, 1))]
    G = G.reshape((N, mask.sum()))
    meanGE = np.nanmean(G, 1)
    res.mean_expression = meanGE

    # initialize permutation
    idx_rand_genes = np.full((NPerm, mask.sum()), np.nan)
    idx_background = _iloc_isMember(ref_ge.symbols, gene.symbols)
    tmpGE = np.full((N, NPerm), np.nan)

    pbar = Progress(NPerm)
    # permutation
    for k in range(NPerm):
        pbar.progress()
        rid: np.ndarray = idx_background[np.random.permutation(idx_background.size)[:mask.sum()]]
        idx_rand_genes[k, :] = rid
        tmpGE[:, k] = np.nanmean(gene.expression[:, rid.astype('int64')], 1)

    pbar.clear()
    res.null_expression = tmpGE
    res.difference = res.mean_expression - np.nanmean(res.null_expression, 1)

    # ===== compute p-value =====
    res.p = np.full((N, 1), np.nan)
    for i in range(N):
        P = np.count_nonzero((tmpGE[i, :] > meanGE[i])) / NPerm
        if P > 0.5:
            res.p[i, 0] = (1 - P) * 2
        else:
            res.p[i, 0] = P * 2

    res.regionDescriptions = gene.regionDescriptions
    print(" >> finished without errors")
    return res


def permutation_expression_null_coexp(goi, expressions: Optional[np.ndarray] = None,
                                      gene_symbols: Optional[np.ndarray] = None,
                                      regionDesc: Optional[str] = None) -> PermResExp:
    print('Running null-coexp model')

    if expressions is None and gene_symbols is None:
        gene = load_gene_expression(True)
    elif expressions is None or gene_symbols is None:
        raise ValueError("Please provide gene symbols of all genes included in the expression data.")
    else:
        gene = Gene(expressions, gene_symbols, regionDesc)

    mask = _check_goi(goi, gene)

    N, K = gene.expression.shape
    # ========== perform permutation ==========
    res = PermResExp()
    NPerm = 1000

    # row mean gene expression
    # G = expressions[:, mask]
    G = gene.expression[np.tile(mask.T, (N, 1))]
    G = G.reshape((N, mask.sum()))

    meanGE = np.nanmean(G, 1)
    res.mean_expression = meanGE

    # compute coexpression of the input GOI
    # coexp_mat = np.corrcoef(G, rowvar=False)
    coexp_mat = np.ma.corrcoef(np.ma.masked_invalid(G), rowvar=False).data
    mask_tri = np.tril(np.ones(coexp_mat.shape), -1)
    coexp = np.nanmean(coexp_mat[mask_tri == 1])
    res.coexp_mean = coexp

    coexp_null = np.full((NPerm, 1), np.nan)
    idx_rand_genes = np.full((NPerm, mask.sum()), np.nan)
    tmpGE = np.full((N, NPerm), np.nan)

    pbar = Progress(NPerm)
    for k in range(NPerm):
        tmp_status = False
        rid = None
        while tmp_status is not True:
            rid, coexp_null[k], tmp_status = _y_rand_gs_coexp(gene.expression, coexp, np.count_nonzero(mask))
        pbar.progress()

        idx_rand_genes[k, :] = rid
        tmpGE[:, k] = np.nanmean(gene.expression[:, rid], 1)

    pbar.clear()
    res.permut_gene_idx = idx_rand_genes
    res.permut_coexp_mean = coexp_null
    res.null_expressions = tmpGE
    res.difference = res.mean_expression - np.nanmean(res.null_expressions, 1)

    # compute p-value
    res.p = np.full((N, 1), np.nan)
    for i in range(N):
        P = np.count_nonzero((tmpGE[i, :] > meanGE[i])) / NPerm
        if P > 0.5:
            res.p[i, 0] = (1 - P) * 2
        else:
            res.p[i, 0] = P * 2

    res.regionDescriptions = gene.regionDescriptions

    print(" >> finished without errors")
    return res


def _check_goi(goi, gene: Gene) -> np.ndarray:
    """
    :return: mask on goi with gene.symbols
    """
    N, K = gene.expression.shape
    print("##", N, "genes detected totally,", K, "brain regions detected.")

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
