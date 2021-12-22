import scipy.sparse
import pickle
import gzip
import pandas as pd
import numpy as np
import scipy.io
import os, sys, re
import logging

def _load_items(dirname, **kwargs):
    name = kwargs.get('name')
    column = kwargs.get('column', -1)
    trim_suffix = kwargs.get('trim', False)
    fbz = os.path.join(dirname, f'{name}.tsv.gz')
    fb = os.path.join(dirname, f'{name}.tsv')
    items = []
    if os.path.exists(fbz):
        with gzip.open(fbz) as fi:
            for line in fi:
                items.append(line.decode('utf-8').strip())
    else:
        with open(fb) as fi:
            for line in fi:
                items.append(line.strip())
    if column >= 0:
        data = []
        for line in items:
            data.append(line.split('\t')[column])
        items = data
    if trim_suffix:
        data = []
        for line in items:
            data.append(re.split('\\W', line)[0])
        items = data
    return items

def load_barcodes(dirname, **kwargs):
    """Load barcodes.tsv or barcodes.tsv.gz"""
    kwargs['name'] = 'barcodes'
    return _load_items(dirname, **kwargs)

def load_features(dirname, **kwargs):
    kwargs['name'] = 'features'
    return _load_items(dirname, **kwargs)

def load_sparse_matrix(dirname:str, **kwargs):
    """Load matrx.mtx
    """
    import gzip
    fm = os.path.join(dirname, 'matrix.mtx')
    mtz = os.path.join(dirname, 'matrix.mtx.gz')

    if os.path.exists(mtz):
        mtx = scipy.io.mmread(mtz)
    elif os.path.exists(fm):
        mtx = scipy.io.mmread(fm)
    else:
        raise Exception('{} does not include data'.format(dirname))

    return mtx

def load_reads_from_sparse_matrix(srcdir:str, **kwargs)->pd.DataFrame:
    verbose = kwargs.get('verbose', False)
    fn_cache = os.path.join(srcdir, '.count.cache')
    if os.path.exists(fn_cache) and os.path.getsize(fn_cache) > 1000:
        df = pd.read_csv(fn_cache, sep='\t', dtype=np.int32)
        return df
    mtx = load_sparse_matrix(srcdir)
    s = np.sum(mtx, axis=0).tolist()[0]
    mtx[mtx>1] = 1
    t = np.sum(mtx, axis=0).tolist()[0]
    del mtx
    df = pd.DataFrame([s, t], columns=['n_Reads', 'n_Features'], index=barcodes).T
    df.to_csv(fn_cache, sep='\t')
    return df
    # return counts_per_cell, features_per_cell

def load_tpm(filename:str, **kwargs):
    output = kwargs.get('output', None)
    use_log = kwargs.get('use_log', None)
    verbose = kwargs.get('verbose', False)
    forced = kwargs.get('forced', False)

    if output is None:
        if use_log:
            output = os.path.join(os.path.dirname(filename), '.{}.logtpm'.format(os.path.basename(filename)))
        else:
            output = os.path.join(os.path.dirname(filename), '.{}.tpm'.format(os.path.basename(filename)))
    if filename.endswith('.tpm') or (os.path.exists(output) and os.path.getsize(output) > 1000):
        if not forced:
            return output
    if verbose:
        sys.stderr.write('converting counts to TPM {}\n'.format(output))
    t = pd.read_csv(filename, index_col=0, sep='\t')
    tpm = t / np.array(np.sum(t, axis=0)) * 1e6
    if use_log:
        tpm = np.log2(tpm + 1)
    tpm.to_csv(output, sep='\t', float_format='%.2f')
    return output

def save_sparse_matrix(dstdir, matrix, barcodes, features, verbose=False):
    import io
    import subprocess
    fn_barcode = os.path.join(dstdir, 'barcodes.tsv.gz')
    fn_mtx = os.path.join(dstdir, 'matrix.mtx')
    fn_features = os.path.join(dstdir, 'features.tsv.gz')

    with io.TextIOWrapper(gzip.open(fn_barcode, 'wb'), encoding='utf-8') as f1,\
        io.TextIOWrapper(gzip.open(fn_features, 'wb'), encoding='utf-8') as f2:
        for c in barcodes:
            f1.write(c + '\n')
        for g in features:
            f2.write(g + '\n')
        if matrix is not None:
            if verbose:
                sys.stderr.write('saving sparse matrix to {}\n'.format(fn_mtx))
            scipy.io.mmwrite(fn_mtx, matrix)
            fn_mtx_gz = fn_mtx + '.gz'
            if os.path.exists(fn_mtx_gz): os.unlink(fn_mtx_gz)
            cmd = 'pigz', '-p', '4', fn_mtx
            if verbose:
                sys.stderr.write('compressing {}\n'.format(fn_mtx))
            proc = subprocess.Popen(cmd).wait()
    pass

def convert_sparse_matrix_to_count(srcdir:str, filename:str=None, **kwargs):
    """Convert sparse matrix to tsv"""
    verbose = kwargs.get('verbose', False)
    forced = kwargs.get('forced', False)
    filename = kwargs.get('filename', None)
    feature_field = kwargs.get('field', None)

    if filename is None:
        filename = os.path.join(srcdir, 'count.tsv')
        if feature_field is not None:
            filename = os.path.join(srcdir, 'count.{}.tsv'.format(feature_field))

    if os.path.exists(filename) and os.path.getsize(filename) > 0 and not forced:
        return filename
    if verbose:
        sys.stderr.write('\033[Kloading barcodes\r')
    barcodes = load_barcodes(srcdir)
    if verbose:
        sys.stderr.write('\033[Kloading features\r')
    features = load_features(srcdir)
    if feature_field is not None:
        f2i = {}
        n_features = 0
        genes = []
        for i, feature in enumerate(features):
            items = feature.split('\t')
            gene = items[feature_field]
            if gene not in f2i: f2i[gene] = []
            genes.append(gene)
            f2i[gene].append(i)
            n_features += 1
        if n_features != len(f2i):
            if verbose:
                sys.stderr.write('degeneration required\n')
        else:
            fetures = genes
            f2i = None
    else:
        f2i = None

    if verbose:
        sys.stderr.write('\033[Kloading sparse matrix\r')
    mtx = load_sparse_matrix(srcdir)
    if f2i is not None:
        if verbose:
            sys.stderr.write('\033[Kaggregating {} features into {} rows\r'.format(mtx.shape[0], len(f2i)))
        mtx = mtx.tocsr()
        rows = []
        ft_ = []
        for gene in sorted(f2i.keys()):
            idx = f2i[gene]
            ft_.append(gene)
            if len(idx) == 1:
                row = mtx[idx[0], :]
            else:
                row = scipy.sparse.csr_matrix(np.sum(mtx[idx, :], axis=0))
            rows.append(row)
        mtx = scipy.sparse.vstack(rows)
        features = ft_

    if verbose:
        sys.stderr.write('\033[Ksaving count matrix to {}\n'.format(filename))
    df = pd.DataFrame(mtx.toarray(), columns=barcodes, index=features, dtype=np.int32)
    df.to_csv(filename, sep='\t')
    return filename

def _check_sparse_matrix(srcdir):
    for f in 'matrix.mtx', 'barcodes.tsv', 'features.tsv':
        flag = False
        for z in ('', '.gz'):
            fn = os.path.join(srcdir, f + z)
            if os.path.exists(fn) and os.path.getsize(fn) > 10:
                flag = True
                break
        if not flag:
            return False
    return True

def load_gene_expression(filename:str, genes:list, **kwargs)->pd.DataFrame:
    """Load expression DataFrame from count/tpm file or sparse matrix.
    If the genes are cached, this function returns DataFrame first.
    """
    import hashlib, sqlite3, gzip, json
    verbose = kwargs.get('verbose', False)
    forced = kwargs.get('forced', False)
    logger = kwargs.get('logger', None)
    if logger is None:
        logger = logging.getLogger(__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)

    # print(filename)
    # print(filename, os.path.isdir(filename), _check_sparse_matrix(filename))
    if os.path.isdir(filename):
        logger.info('loading sparse matrix')
        if _check_sparse_matrix(filename):
            feature_field = kwargs.get('feature_field', 1)
            filename = convert_sparse_matrix_to_count(filename, field=feature_field, verbose=verbose, forced=forced)
        else:
            raise Exception("invalid directory")

    srcdir = os.path.dirname(filename)

    genes = set(genes)
    if isinstance(genes, str):
        genes = [genes, ]
    filename_db = os.path.join(srcdir, '.' + os.path.basename(filename) + '.expr.db')
    
    logger.info('expression database file is {}'.format(filename_db))
    expression = {}
    with sqlite3.connect(filename_db) as cnx:
        cur = cnx.cursor()
        cur.execute('create table if not exists expression(gene not null primary key, data blob)')
        cached_genes = []
        genes_loaded = []
        if not forced:
            cstr = ''
            if len(genes) < 250:
                for gene in genes:
                    if cstr != '':
                        cstr += ' or '
                    cstr += 'gene="{}"'.format(gene)
                sqlcom = 'select gene, data from expression where ' + cstr
            else:
                sqlcom = 'select gene, data from expression'
            if verbose:
                sys.stderr.write(sqlcom + '\n')
            cur.execute(sqlcom)
            for r in cur.fetchall():
                gene = r[0]
                if gene in genes:
                    if r[1] is not None and len(r[1]) == 0:
                        logger.info(f'reading {gene} from cache')
                        values = json.loads(gzip.decompress(r[1]).decode('utf-8'))
                        if len(values) > 0:
                            expression[gene] = values
                    else:
                        genes_loaded.append(gene)
        n_genes = len(genes)

        existing = set()
        cur.execute('select gene from expression')
        for r in cur.fetchall():
            existing.add(r[0])
        
        if len(expression) == n_genes:
            with open(filename) as fi:
                columns = [x_.strip('"\n') for x_ in fi.readline().split('\t')]
                if columns[0] != '' and (columns[0].lower() not in ('tracking_id', 'gene', 'transcript_id', 'gene_id')):
                    columns = columns
                else:
                    columns = columns[1:]
            m_ = []
            g_ = []
            for g in sorted(expression.keys()):
                g_.append(g)
                m_.append(expression[g])
            return pd.DataFrame(np.array(m_, dtype=np.float32), index=g_, columns=columns)
        genes_to_load = set([g for g in genes if g not in genes_loaded])
        genes_to_save = []
        logger.info('{}'.format(','.join(genes_to_load)))
        logger.info('loading {} genes from {} : {}'.format(len(genes_to_load), filename, ','.join(genes_to_load)))
        with open(filename) as fi:
            columns = [x_.strip('"\n') for x_ in fi.readline().split('\t')]
            if columns[0] != '' and (columns[0].lower() not in ('tracking_id', 'gene', 'transcript_id', 'gene_id')):
                columns = columns
            else:
                columns = columns[1:]

            for line in fi:
                gene, row = line[0:-1].split('\t', 1)
                gene = gene.strip('"')
                if gene in genes:
                    values = [float(x_) for x_ in row.split('\t')]
                    expression[gene] = values
                    genes_to_save.append(gene)
        for gene in genes_to_save:
            e_ = expression[gene]
            valueobj = gzip.compress(json.dumps(e_).encode('utf-8'))
            if gene in existing:
                cur.execute('update expression set data=? where gene=?', (valueobj, gene))
            else:
                cur.execute('insert into expression (gene, data) values(?, ?)', (gene, valueobj))
        for gene in genes_to_load:
            if gene not in genes_to_save:
                # zobj = gzip.compress('[]'.encode('utf-8'))
                logger.info(f'{gene} is set as NULL')
                # logger.info(zobj)
                cur.execute('insert into expression (gene, data) values(?, NULL)', (gene, ))
        m_ = []
        g_ = []
        for g in sorted(expression.keys()):
            g_.append(g)
            m_.append(expression[g])
        return pd.DataFrame(np.array(m_, dtype=np.float32), index=g_, columns=columns)

