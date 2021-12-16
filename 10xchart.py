"""
Visualization script used in Ohinata et al. 2021.
To detect clusters of dead cells or those of multiplet cells, 
'count' command was applied and read count distribution was estimated.
Contaminated feeder cells weere removed by gene expression specific to 
them such as Ctcf using 'marker' command.
THe command 'cluster' displays colored clusters in 2d plot.
"""

import os, sys, re
import pandas as pd
import numpy as np
import scipy.io
import argparse
import umap
import plotly
import plotly.graph_objs as go
import plotly.subplots
import scipy.sparse
import pickle
import gzip
from cellrangerwrapper import *

def calculate_coordinate_color(pos, cmin, cmax, random_factor=0.4, colormode=0):
    deg = [(pos[i] - cmin[i])/(cmax[i] - cmin[i]) for i in range(len(pos))]
    print(pos, cmin, cmax, deg)
    if len(deg) == 2:
        if colormode == 0:
            red = (1-deg[1])*(1.5-deg[0])
            blue = deg[0]
            green = deg[1] * .8
        elif colormode == 1:
            red = 1 - deg[0]
            green = 1 - deg[1]
            blue = deg[1]
        elif colormode == 2:
            red = deg[0]
            blue = (1-deg[0]) * (1-deg[1])
            green = deg[1]
        elif colormode == 3:
            red = (0.5-deg[0]) * 2 + deg[1] * .05
            green = (deg[0] - .5) * 2 - deg[1] * .1
            blue = 1 - abs(deg[0] - 0.5) * 4 - deg[1] * .15
        elif colormode == 4:
            red = (1-deg[0]) * (1-deg[1])
            blue = deg[0] * (1-deg[1])
            green = deg[1] - deg[0] * .1
        elif colormode == 5:
            red = deg[0] * deg[1]
            blue = 1 - deg[1] + deg[0] * .05
            green = (1-deg[0]) ** 2 + deg[1] * .1
        elif colormode == 6:
            green = deg[0] + (deg[1] - .6) * 2.5
            red = deg[0] * (1-deg[1])
            blue = (1-deg[0]) ** 2 - deg[1] * .1
        else:
            return None        
    else:
        red = deg[0]
        blue = deg[1]
        green = deg[2]
    values = np.array((red, green, blue))
    if random_factor > 0:
        values += (np.random.rand(3) - 0.5) * random_factor 
    rgb = np.array(values * 256, dtype=np.int32)
    rgb[rgb<0] = 0
    rgb[rgb>255] = 255
    return 'rgb({},{},{})'.format(rgb[0], rgb[1], rgb[2])

def load_reads_from_sparse_matrix(srcdir:str, **kwargs)->pd.DataFrame:
    verbose = kwargs.get('verbose', False)
    fn_cache = os.path.join(srcdir, '.count.cache')
    if os.path.exists(fn_cache) and os.path.getsize(fn_cache) > 1000:
        # print(fn_cache)
        df = pd.read_csv(fn_cache, sep='\t', index_col=0)
        return df
    mtx = load_sparse_matrix(srcdir)
    s = np.array(np.sum(mtx, axis=0).reshape(-1, 1))
    m = mtx.toarray()
    del mtx
    m[m>1] = 1
    t = np.sum(m, axis=0).reshape(-1, 1)
    del m
    barcodes = load_barcodes(srcdir)
    df = pd.DataFrame(np.concatenate([s, t], axis=1), columns=['n_Reads', 'n_Features'], index=barcodes)
    df.to_csv(fn_cache, sep='\t')
    return df
    # return counts_per_cell, features_per_cell

def _load_sparse_matrix(dirname:str, **kwargs):
    """Load matrx.mtx
    """
    import gzip
    # fgz = os.path.join(dirname, 'features.tsv.gz')
    fm = os.path.join(dirname, 'matrix.mtx')
    mtz = os.path.join(dirname, 'matrix.mtx.gz')

    if os.path.exists(fm):
        mtx = scipy.io.mmread(fm)
    elif os.path.exists(mtz):
        mtx = scipy.io.mmread(mtz)
    else:
        raise Exception('{} does not include data'.format(dirname))

    return mtx


def show_marker(arguments=None):
    """
    return {'scatter.html':fn_scatter,
        'info':fn_info,
        'violin.html':fn_violin,
        'total.html':fn_total   
    }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help='expression tsv')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-g', nargs='+')
    parser.add_argument('-c', help='cluster')
    parser.add_argument('-u', help='coordinates such as UMAP or tSNE')
    parser.add_argument('-d', type=int, default=2, choices=[2, 3], help='2D or 3D chart')
    parser.add_argument('-o', default='exprchart')
    parser.add_argument('--threshold', default=1, type=float, help='expression threshold, 0.9 as 90% in percentile mode')
    parser.add_argument('--percentile', action='store_true')
    parser.add_argument('--output-all', action='store_true')
    args = parser.parse_known_args(arguments)[0]

    verbose = args.verbose

    chart_type = args.d
    fn_input = list(sorted(args.u))[0]#, key=lambda f:os.))[0]

    outdir = args.o
    mode2d = (args.d == 2)
    percentile_mode = args.percentile
    fn_expression = args.e
    fn_coord = args.u
    output_all = args.output_all
    threshold = args.threshold

    os.makedirs(outdir, exist_ok=True)

    fn_info = os.path.join(outdir, 'run.marker.info')
    fn_scatter = os.path.join(outdir, 'markerexpression.html')
    fn_violin = os.path.join(outdir, 'violin.html')
    fn_total = os.path.join(outdir, 'clusters.html')

    if verbose: sys.stderr.write('loading coordinates from {}\n'.format(fn_coord))
    coord = pd.read_csv(fn_coord, sep='\t', index_col=0)

    if args.c:
        clusterdf = pd.read_csv(args.c, sep='\t', index_col=0)
        if 'Cluster' in clusterdf.columns:
            clusters = clusterdf['Cluster'].values
        else:
            clusters = clusterdf.values[:,0]
        if verbose:
            sys.stderr.write('Cluster data loaded from {}\n'.format(args.c))
        del clusterdf
        n_clusters = max(clusters) + 1
    else:
        clusters = None
        n_clusters = 0
        clusters = []
        cluster_separator = '_'
        cluster_groups = {}
        for index in coord.index:
            group = index.split(cluster_separator)[0]
            if group not in cluster_groups:
                cluster_groups[group] = len(cluster_groups) 
            clusters.append(cluster_groups[group])
            
    markers = []
    for gene in args.g:
        if os.path.exists(gene):
            with open(gene) as fi:
                for line in fi:
                    items = re.split('[\\s,]+')
                    if len(items[0]) > 0:
                        markers.append(items[0])
        else:
            markers.append(gene)
    if verbose: sys.stderr.write('loading gene expression from {}\n'.format(fn_expression))
    expr = load_gene_expression(fn_expression, markers, verbose=verbose)
    marker_genes = sorted([g for g in markers if g in expr.index])

    if verbose:
        for gene in sorted(markers):
            if gene not in expr.index: continue
            values = expr.loc[gene].values
            pt = np.percentile(values, [0, 50, 90, 95, 99, 99.9, 100])
            sys.stderr.write('{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(gene, pt[0], pt[1], pt[2]))

    cluster2index = {}
    with open(fn_info, 'w') as fo:
        fo.write('input:{}\n'.format(fn_expression))
        fo.write('n_cells:{}\n'.format(coord.shape[0]))
        if clusters is not None:
            fo.write('clusers:{}\n'.format(n_clusters))
            for cn in set(clusters):
                index = [j for j in range(len(clusters)) if clusters[j] == cn]
                n_cells_cluster = len(index)
                if n_cells_cluster > 0:
                    cluster2index[cn] = index
                    fo.write('cluster_C{}={} cells\n'.format(cn, n_cells_cluster))
        if verbose:
            for i, marker in enumerate(marker_genes):
                fo.write('marker{}:{}\t{}\n'.format(i + 1, marker, marker in expr.index))

    n_cols = (expr.shape[0] + 7) // 8
    n_rows = min(8, expr.shape[0])# // 8
    symbols = ['circle', 'diamond', 'square', 'triangle-up', 'circle-open-dot', 
    'diamond-open-dot', 'square-open-dot', 'cross', 'triangle-left', 'triangle-left-open']
    symbols = ['circle', ]

    if clusters is not None: #
        # Violin chart
        n_charts = len(marker_genes)
        fig = plotly.subplots.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=marker_genes)
        fig.print_grid()
        traces = []
        for i, g in enumerate(marker_genes):
            values = expr.loc[g].values
            row = (i % n_rows) + 1
            col = (i // n_rows) + 1
            fig.add_trace(go.Violin(x=clusters, y=values, name=g, meanline_visible=True, opacity=0.6, box_visible=False), 
            row=row, col=col)

        fig.update_xaxes(title='Cluster', linewidth=1, linecolor='black', showgrid=False, showline=True, mirror='ticks')
        fig.update_yaxes(title='Expression', linewidth=1, linecolor='black', showgrid=False, showline=True, mirror='ticks')
        fig.update_layout(plot_bgcolor='white')
        plotly.offline.plot(fig, filename=fn_violin, auto_open=False)
        # cluseter chart
        # color chart 
        # ES red
        # TS green
        # PrES blue
        comin = np.min(coord, axis=0)
        comax = np.max(coord, axis=0)
        xmin, xmax, ymin, ymax = comin[0], comax[0], comin[1], comax[1]

        if output_all:
            traces = []
            i_ = 0
            if mode2d:
                layout = go.Layout(
                    xaxis=dict(title=coord.columns[0], linewidth=1, linecolor='black', showgrid=False, showline=True, mirror='ticks'), 
                    yaxis=dict(title=coord.columns[1], linewidth=1, linecolor='black', showgrid=False, showline=True, mirror='ticks'),
                    plot_bgcolor='white'            
                )
            else:
                layout = go.Layout()
            for cn in sorted(cluster2index.keys()):
                index = cluster2index[cn]
                name = 'C{} ({})'.format(cn, len(index))
                xyz = coord.values[index,:]
                if cn < 0:
                    name = 'Unclassified'
                    color = 'lightgray'
                else:
                    color = None
                texts = expr.columns[index]
                if mode2d:
                    trace = go.Scattergl(x=xyz[:,0], y=xyz[:,1], name=name, mode='markers',
                        text=texts,
                        marker=dict(size=4, color=color,
                        symbol=symbols[i_ % len(symbols)]
                        ))
                else:
                    trace = go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], 
                        name=name, mode='markers', text=texts,
                        marker=dict(size=4, color=color, symbol=symbols[i_ % len(symbols)])
                    )
                traces.append(trace)
                i_ += 1
            fig = go.Figure(traces, layout)
            plotly.offline.plot(fig, fn_total)

    traces = []
    xyz = coord.values


    if mode2d:
        # print(xyz.shape)
        traces.append(go.Scattergl(x=xyz[:,0], y=xyz[:,1], mode='markers',text=list(coord.index),
            marker=dict(size=3, color='lightgray'), name='{} cells'.format(coord.shape[0])))
            
        layout = go.Layout(
            xaxis=dict(title=coord.columns[0], linewidth=1, linecolor='black', showgrid=False, showline=True, mirror='ticks'), 
            yaxis=dict(title=coord.columns[1], linewidth=1, linecolor='black', showgrid=False, showline=True, mirror='ticks'),
            plot_bgcolor='white'            
        )
    else:
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title=coord.columns[0]), 
                yaxis=dict(title=coord.columns[1]),
                zaxis=dict(title=coord.columns[2])
            )
        )
        raise Exception('no 3d')


    for i, marker in enumerate(marker_genes):
        if marker not in expr.index: continue
        values = expr.loc[marker].values
        if percentile_mode:
            pt = np.percentile(values, [0, threshold])
            thr_value = pt[1]
        else:
            thr_value = threshold
        index = []
        for j, v in enumerate(values):
            if v > thr_value:
                index.append(j)
        if len(index) == 0:
            continue
        if verbose and percentile_mode:
            # print(pt, np.percentile(values, [0,10,25,50,75,90,100]))
            sys.stderr.write('{}\t{:.1f}\t{:.1f}\t{}\n'.format(marker, threshold, thr_value, len(index)))
        xyz = coord.values[index]
        texts = coord.index[index]
        if mode2d:
            # print(texts[0:10])
            traces.append(go.Scattergl(x=xyz[:,0], y=xyz[:,1], mode='markers',
            name=marker,text=list(texts),
            marker=dict(size=4, symbol=symbols[i % len(symbols)])))
    fig = go.Figure(traces, layout)
    plotly.offline.plot(fig, filename=fn_scatter)

    return {'scatter.html':fn_scatter,
        'info':fn_info,
        'violin.html':fn_violin,
        'total.html':fn_total   
    }

def display_cluster_map(arguments=None):
    """Cluster coodinates
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help='expression tsv')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-g', nargs='+')
    parser.add_argument('-c', help='cluster')
    parser.add_argument('--color', metavar='tsv file', help='Color map of each cluster')
    parser.add_argument('-u', nargs='+', help='coordinates such as UMAP or tSNE')
    parser.add_argument('-d', type=int, default=2, choices=[2, 3], help='2D or 3D chart')
    parser.add_argument('-o', default='exprchart')
    parser.add_argument('-t', default=1, type=float, help='expression threshold')
    parser.add_argument('--percentile', action='store_true')
    parser.add_argument('--colormode', type=int, default=-1, help='color mode')
    parser.add_argument('--preset-color', action='store_true')
    parser.add_argument('--min-cluster-size', default=10, type=int)
    args = parser.parse_known_args(arguments)[0]

    verbose = args.verbose
    chart_type = args.d
    fn_input = list(sorted(args.u))[0]
    outdir = args.o
    mode2d = (args.d == 2)
    percentile_mode = args.percentile
    fn_expression = args.e

    os.makedirs(outdir, exist_ok=True)
    lower_limit = args.min_cluster_size

    fn_info = os.path.join(outdir, 'run.info')
    fn_scatter = os.path.join(outdir, 'scatter.html')
    fn_violin = os.path.join(outdir, 'violin.html')
    fn_total = os.path.join(outdir, 'clusters.html')

    marker_group = [
        ['Pou5f1', 'Nanog', 'Sox2', 'Zfp42', ], # ES red
        ['Apoe','Apoa1','Igf2','Ttr','Amn','Cldn6','Dab2','Cyp26a1','Krt8','Krt18','Krt19 Foxa2','Vegfa','Hnf1b','Acvr1','Gata4','Gata6','Afp','Hnf4a','Cited1','Rhox5','Ihh','Furin','Sox17','Sox7','Bmp2','Foxh1','Fgf5','Fgf8','Otx2','Nodal','Thbd','Stra6','Pdgfra','Fst','Sparc','Plat','Pth1r','Cubn'], # PrES green
        ['Cdx2','Eomes','Sox2','Esrrb','Elf5','Prl3d1','Csh1','Ascl2','Tpbpa','Esx1','Dlx3'] # TS blue
    ]

    coord = pd.read_csv(fn_input, sep='\t', index_col=0)
    cluster2color = None

    if args.color is not None:
        cluster2color = {}
        with open(args.color) as fi:
            for line in fi:
                items = line.strip().split('\t')
                if len(items) > 1 and re.match('\\-?\\d+$', items[0]):
                    cluster2color[int(items[0])] = items[1]
        colormode = -1
    else:
        colormode = args.colormode        

    # lower_limit = 100
    alt_labels = None
    # presets = ['all', 'pres']

    if args.c is not None and os.path.exists(args.c):
        if os.path.exists(args.c):
            clusterdf = pd.read_csv(args.c, sep='\t', index_col=0)
        else:
            clusterdf = None

        # print(clusterdf)
        if 'Cluster' in clusterdf.columns:
            clusters = clusterdf['Cluster'].values
        else:
            clusters = clusterdf.values[:,0]
        print(clusters)
        cluster2color = {}
        if verbose:
            sys.stderr.write('Cluster data loaded from {}\n'.format(args.c))
        del clusterdf
    else:
        clusters = None
    use_preset_color = args.preset_color
    alt_labels = None
    if use_preset_color:
        # if clusters is None:
        if clusters is None:
            # clusters = None
            clusters = []
            n_clusters = 0
            cluster_separator = '.'
            cluster_groups = {}
            # cluster2color = {}
            for index in coord.index:
                group = index.split(cluster_separator)[0]
                if group not in cluster_groups:
                    cluster_groups[group] = len(cluster_groups) 
                clusters.append(cluster_groups[group])
                # cluster2color[cluster_groups[group]] = None
            cc = [None] * 27
            tmp_celltype = \
                """p0, t0, p0, e0, t0,   e0, p0, p0, t0, t3, 
                t3, t3, p3, e3, e3,   p3, p3, t3, t6, p6, 
                t6, e6, t6, p6, p6,   t6, p6, p6, p6""".split(',')
        else:
            tmp_celltype = ['p6', 'p0', 'p3', 'p6', 'p0', 'p6', 'p3', 'p6']
            n_clusters = len(tmp_celltype)
            # for i in range(n_clusters):
            #     clusters.append(i)
            cc = [None] * n_clusters
            print(clusters)
        alt_labels = []
        # color2cluster = []
        cluster2color = {}
        used = set()
        for cn, tc in enumerate(tmp_celltype):
            tc = tc.strip()
            if tc.startswith('p'):
                red = green = 16
                blue = 128
                celltype = 'PrES'
            elif tc.startswith('e'):
                red = 128
                blue = green = 16
                celltype = 'ES'
            else:
                green = 100
                red = blue = 32
                celltype = 'TS'
            alt_labels.append('{} day{}'.format(celltype, tc[-1]))
            if tc.endswith('3'):
                red *= 2
                blue *= 2
                green *= 2
            elif tc.endswith('6'):
                red += 127
                blue += 127
                green += 127
            if tc in used:
                red += (np.random.randint(0, 64) - 32)
                green += (np.random.randint(0, 64) - 32)
                blue += (np.random.randint(0, 64) - 32)
            else:
                used.add(tc)
            print(tc, red, green, blue)

            red = max(0, min(255, red))
            green = max(0, min(255, green))
            blue = max(0, min(255, blue))

            cluster2color[cn] = 'rgb({},{},{})'.format(red, green, blue)

    n_clusters = max(clusters) + 1
    cluster2index = {}
    n_clusters = 0
    def __cluster_order(cn):
        if alt_labels is not None:
            return alt_labels[cn]
        else:
            return cn
    for cn in sorted(clusters, key=__cluster_order):
        index = [j for j in range(len(clusters)) if clusters[j] == cn]
        n_cells_cluster = len(index)#np.count_nonzero(clusters[clusters==i])
        if n_cells_cluster > 0:
            cluster2index[cn] = index
            if cn >= 0:
                n_clusters += 1
    traces = []
    i_ = 0

    symbols = ['circle', 'diamond', 'rectangle', 'triangle-up', 'circle-open', 
    'diamond-open', 'rectangle-open', 'cross']
    symbols = ['circle', ]

    cogtable = []
    for cn, index in cluster2index.items():
        xyz = coord.values[index,0:2]
        cogtable.append(np.mean(xyz, axis=0))
    cmin = np.min(np.array(cogtable), axis=0)
    cmax = np.max(np.array(cogtable), axis=0)
    if mode2d:
        cmin = cmin[0:2]
        cmax = cmax[0:2]
    else:
        cmin = cmin[0:3]
        cmax = cmax[0:3]

    for cn, index in cluster2index.items():
        if cn < 0: continue
        if lower_limit > 0 and len(index) < lower_limit: continue
        if mode2d:
            xyz = coord.values[index,0:2]
        else:
            xyz = coord.values[index,0:3]
        name = 'C{} ({})'.format(cn + 1, len(index))
        if alt_labels is not None:
            name = alt_labels[cn]
        texts = coord.index[index]
        marker = dict(size=2, symbol=symbols[i_ % len(symbols)])
        if cn in cluster2color:
            print(cn, cluster2color[cn])
            marker['color'] = cluster2color[cn]
        else:
            rgb = calculate_coordinate_color(np.mean(xyz, axis=0), 
                cmin, cmax, colormode=colormode)
            if rgb is not None:
                marker['color'] = rgb
        if mode2d:
            trace = go.Scattergl(x=xyz[:,0], y=xyz[:,1], name=name, text=texts,
            mode='markers', marker=marker)
        else:
            trace = go.Scatter3d(x=xyz[:,0], y=xyz[:,1], z=xyz[:,2], name=name,
                text=texts, mode='markers', marker=marker)
        traces.append(trace)
    if colormode < 0:    
        layout = go.Layout(xaxis=dict(title=coord.columns[0]), 
            yaxis=dict(title=coord.columns[1]),
            plot_bgcolor='white')
    else:
        layout = go.Layout(xaxis=dict(title=coord.columns[0]), 
            yaxis=dict(title=coord.columns[1]),
            plot_bgcolor='white')
    fig = go.Figure(traces, layout)
    fig.update_xaxes(linewidth=1, showline=True, mirror='ticks', linecolor='black', 
        title=coord.columns[0], showgrid=False)
    fig.update_yaxes(linewidth=1, showline=True, mirror='ticks', linecolor='black',
        title=coord.columns[1], showgrid=False)
    plotly.offline.plot(fig, filename=fn_scatter)
        

def count_tags_by_cluster(arguments=None):
    """Draw count distribution charts by cluster

    return
    outputs = {
        'count.tsv':fn_count,
        'stats.tsv':fn_stat,
        'color.tsv':fn_color,
        'depth.graph.html':fn_depth,
        'count.graph.html':fn_histogram
    }
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', metavar='TSV file', help='cluster TSV file')
    parser.add_argument('-i', metavar='directory', help='matrix directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--forced', action='store_true')
    parser.add_argument('-o', default='stat')
    parser.add_argument('-e', default=None)
    parser.add_argument('--bin', type=int, default=50)
    args = parser.parse_known_args(arguments)[0]

    verbose = args.verbose
    srcdir = args.i

    # set filenames of output
    outdir = args.o
    os.makedirs(outdir, exist_ok=True)
    fn_count = os.path.join(outdir, 'count_by_cluster.tsv')
    fn_stat = os.path.join(outdir, 'count_stat.tsv')
    fn_color = os.path.join(outdir, 'cluster_color.tsv')
    fn_depth = os.path.join(outdir, 'depth.html')
    fn_histogram = os.path.join(outdir, 'count_by_cluster.html')
    fn_expr = args.e
    fn_cluster = args.c
    forced = args.forced

    if os.path.exists(fn_stat) is False or forced:
        if verbose: sys.stderr.write('\033[Kloading clusters\r')
        clusterdf = pd.read_csv(fn_cluster, sep='\t', index_col=0)
        if 'Cluster' in clusterdf.columns:
            clusters = clusterdf['Cluster'].values
        else:
            clusters = clusterdf.values[:,0]
        n_clusters = max(clusters) + 1
        n_cells = clusterdf.shape[0]
        fn_out = args.o
        if verbose: sys.stderr.write('\033[K{} clusters of {} cells loaded\n'.format(n_clusters, n_cells))

        if verbose:
            sys.stderr.write('\033[Kloading barcodes\r')
        barcodes = load_barcodes(srcdir)
        barcode2column = {}
        for i, b in enumerate(barcodes):
            barcode2column[b] = i
        cluster2bcd = {}
        for cn in range(n_clusters):
            cluster2bcd[cn] = []    
        for i, barcode in enumerate(clusterdf.index):
            if clusters[i] >= 0:
                cluster2bcd[clusters[i]].append(barcode2column[barcode])
        if verbose:
            sys.stderr.write('\033[K{} barcodes loaded\n'.format(len(barcodes)))

        if verbose:
            sys.stderr.write('\033[Kloading matrix\r')
        cdf = load_reads_from_sparse_matrix(srcdir)
        counts_per_cell = cdf['n_Reads'].values.reshape(-1)
        features_per_cell = cdf['n_Features'].values.reshape(-1)
        if verbose:
            sys.stderr.write('\033[K{} x {} matrix loaded\n'.format(cdf.shape[0], cdf.shape[1]))

        n_cells = len(barcodes)
        reads_and_genes = np.zeros((n_cells, 3), dtype=np.float32)
        for i, barcode in enumerate(barcodes):
            col = barcode2column[barcode]
            if barcode in clusterdf.index:
                clus = clusterdf.loc[barcode].values[0]
            else:
                clus = -1
            n_reads = counts_per_cell[i]
            n_features = features_per_cell[i]
            reads_and_genes[i] = (clus, n_reads, n_features)
        stat = pd.DataFrame(reads_and_genes, columns=['Cluster', 'n_reads', 'n_features'], index=barcodes, dtype=np.int32)
        stat.to_csv(fn_stat, sep='\t')
    else:
        stat = pd.read_csv(fn_stat, sep='\t', index_col=0)

    # scatter by cluster
    import collections
    clusters = set(stat['Cluster'])
    cluster_to_data = collections.OrderedDict()

    fn_scatter = os.path.join(outdir, 'coloredclusters.html')
    traces = []
    cluster_title = []

    field_of_interest = 'n_reads' # color
    total = stat[field_of_interest].values
    total_mean = np.mean(total)
    total_sd = np.std(total)
    total_med = np.median(total)
    cluster_color = {}

    for cn in sorted(clusters):
        submat = stat[stat['Cluster']==cn]
        cluster_to_data[cn] = submat
        X = submat['n_reads']
        Y = submat['n_features']

        values = submat[field_of_interest].values
        submat_mean = np.mean(values)
        submat_sd = np.std(values)
        submat_med = np.median(values)
        z = (submat_mean - total_mean) / total_sd
        deg = z
        u = min(255, max(0, int((deg + 1) * 256)))
        d = min(255, max(0, int((1 - deg) * 256)))
        red = 255 if deg > 0 else u
        green = min(u, d)
        blue = 255 if deg < 0 else d
        cluster_color[cn] = ['rgb({:.0f},{:.0f},{:.0f})'.format(red, green, blue), z]
        if cn >= 0:
            title = 'C{} ({})'.format(cn + 1, submat.shape[0])
            cluster_title.append(title)
        else:
            title = 'Unclassified ({})'.format(submat.shape[0])
        traces.append(go.Scatter(x=X, y=Y, 
            mode='markers', 
            marker=dict(size=5),
            name=title))
    # set color
    if cluster_color is not None and len(cluster_color) > 0:
        with open(fn_color, 'w') as fo:
            for cn in sorted(clusters):
                cinfo = cluster_color[cn]
                fo.write('{:.0f}\t{}\t{:.4f}\n'.format(cn, cinfo[0], cinfo[1]))

    fig = go.Figure(traces)
    plotly.offline.plot(fig, filename=fn_depth, auto_open=False)

    binsize = args.bin
    n_clusters = len(cluster_title)#df.shape[1]
    n_chart_cols = 3
    n_chart_rows = (n_clusters + 1 + n_chart_cols - 1) // n_chart_cols
    maxcnt = ((np.max(stat['n_reads']) + binsize - 1) // binsize) * binsize

    fig = plotly.subplots.make_subplots(
        cols=n_chart_cols, 
        rows=n_chart_rows, 
        subplot_titles=cluster_title)

    xlimit = min(30000, maxcnt) #// binsize
    gsize = xlimit // binsize
    x = np.arange(0, maxcnt, binsize)
    accum = np.zeros(x.shape[0], dtype=np.int32)

    index = 0
    for cn in sorted(cluster_to_data.keys()):
        if cn < 0: continue
        row = (index // n_chart_cols) + 1
        col = (index % n_chart_cols) + 1
        y = np.zeros(x.size, dtype=np.int32)
        submat = cluster_to_data[cn]
        n_reads = submat['n_reads']
        for n in submat['n_reads'].values:
            y[n // binsize] += 1
        accum += y
        trace = go.Scatter(x=x[0:gsize], y=y[0:gsize], name=cluster_title[index])
        fig.add_trace(trace, row=row, col=col)
        index += 1

    trace = go.Scatter(x=x, y=accum, name='Total')
    fig.add_trace(trace, row=n_chart_rows, col=n_chart_cols)

    fig.update_xaxes(range=[0,xlimit])
    fig.update_xaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=False, showline=True, linewidth=1, linecolor='black')
    
    fig.update_layout(legend=None, plot_bgcolor='white')
    plotly.offline.plot(fig, filename=fn_histogram, auto_open=False)

    outputs = {
        'cluter.count.tsv':fn_count,
        'stats.tsv':fn_stat,
        'color.tsv':fn_color,
        'depth.graph.html':fn_depth,
        'count.graph.html':fn_histogram
    }

    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['marker', 'count', 'cluster', 'all'])
    parser.add_argument('--seurat-dir', default=None, metavar='directory', help='output directory of seurat.R script')
    parser.add_argument('-e', help='expression tsv', default=None)
    parser.add_argument('-g', nargs='+', default=None, help='marker genes for marker command')
    parser.add_argument('-u', default=None, help='coordinates such as UMAP or tSNE')
    parser.add_argument('-o', default='analysis', metavar='directory', help='output')

    parser.add_argument('-t', metavar='number', default=1, type=float, help='expression threshold')
    parser.add_argument('--percentile', action='store_true', help='percentile mode')

    parser.add_argument('-c', metavar='filename', help='cluster TSV file')
    parser.add_argument('-i', metavar='directory', help='matrix directory')

    parser.add_argument('--forced', action='store_true', help='force calculation')
    parser.add_argument('--bin', type=int, default=100, metavar='number', help='count command graph bin')


    parser.add_argument('--color', metavar='tsv file', help='Color map of each cluster')

    parser.add_argument('-d', type=int, default=2, choices=[2, 3], help='2D or 3D chart')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_known_args()[0]
    cargs = list(sys.argv[2:])
    cmd = args.cmd
    if args.seurat_dir is not None:
        basedir = args.seurat_dir
        if args.e is None: 
            cargs += ['-e', os.path.join(basedir, 'seurat.normalized.tsv')]
        if args.u is None:
            cargs += ['-u', os.path.join(basedir, 'seurat.umap.tsv')]
        if args.c is None:
            cargs += ['-c', os.path.join(basedir, 'seurat.clusters.tsv')]

    if cmd == 'all':
        if args.g is not None:
            show_marke(cargs)
        results = count_tags_by_cluster(cargs)
        if args.e is None:
            cargs += ['-e', results['count.tsv']]
        if args.g is not None:
            show_marker(cargs)
        cargs += ['--color', results['color.tsv']]
        display_cluster_map(cargs)
    elif cmd == 'marker':
        show_marker(cargs)
    elif cmd == 'count':
        info = count_tags_by_cluster(cargs)
    # outputs = {
    #     'cluter.count.tsv':fn_count,
    #     'stats.tsv':fn_stat,
    #     'color.tsv':fn_color,
    #     'depth.graph.html':fn_depth,
    #     'count.graph.html':fn_histogram
    # }
        fn_coord = args.u
        if fn_coord is not None:
            cargs += ['--color', info['color.tsv']]
            display_cluster_map(cargs)
    elif cmd == 'cluster':
        display_cluster_map(cargs)
    else:
        raise Exception('not implemented {}'.format(cmd))

if __name__ == '__main__':
    main()
