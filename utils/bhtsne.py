#!/usr/bin/env python

from argparse import ArgumentParser, FileType
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from sys import stderr, stdin, stdout
from tempfile import mkdtemp
from platform import system
from os import devnull
import numpy as np

# Constants
IS_WINDOWS = system() == 'Windows'
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'windows', 'bh_tsne.exe') if IS_WINDOWS else path_join(dirname(__file__), 'bh_tsne')
assert isfile(BH_TSNE_BIN_PATH), f'Unable to find the bh_tsne binary in the same directory as this script, have you forgotten to compile it?: {BH_TSNE_BIN_PATH}'

# Default hyper-parameter values from van der Maaten (2014)
DEFAULT_NO_DIMS = 2
INITIAL_DIMENSIONS = 50
DEFAULT_PERPLEXITY = 50
DEFAULT_THETA = 0.5
EMPTY_SEED = -1

def _argparse():
    argparse = ArgumentParser(description='bh_tsne Python wrapper')
    argparse.add_argument('-d', '--no_dims', type=int, default=DEFAULT_NO_DIMS)
    argparse.add_argument('-p', '--perplexity', type=float, default=DEFAULT_PERPLEXITY)
    argparse.add_argument('-t', '--theta', type=float, default=DEFAULT_THETA)  # 0.0 for theta is equivalent to vanilla t-SNE
    argparse.add_argument('-r', '--randseed', type=int, default=EMPTY_SEED)
    argparse.add_argument('-n', '--initial_dims', type=int, default=INITIAL_DIMENSIONS)
    argparse.add_argument('-v', '--verbose', action='store_true')
    argparse.add_argument('-i', '--input', type=FileType('r'), default=stdin)
    argparse.add_argument('-o', '--output', type=FileType('w'), default=stdout)
    return argparse

class TmpDir:
    def __enter__(self):
        self._tmp_dir_path = mkdtemp()
        return self._tmp_dir_path

    def __exit__(self, type, value, traceback):
        rmtree(self._tmp_dir_path)

def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))

def bh_tsne(samples, no_dims=DEFAULT_NO_DIMS, initial_dims=INITIAL_DIMENSIONS, perplexity=DEFAULT_PERPLEXITY, 
            theta=DEFAULT_THETA, randseed=EMPTY_SEED, verbose=False):

    samples -= np.mean(samples, axis=0)
    cov_x = np.dot(np.transpose(samples), samples)
    eig_val, eig_vec = np.linalg.eig(cov_x)

    # sorting the eigen-values in descending order
    eig_vec = eig_vec[:, eig_val.argsort()[::-1]]

    if initial_dims > len(eig_vec):
        initial_dims = len(eig_vec)

    # truncating the eigen-vectors matrix to keep the most important vectors
    eig_vec = eig_vec[:, :initial_dims]
    samples = np.dot(samples, eig_vec)

    sample_dim = len(samples[0])
    sample_count = len(samples)

    with TmpDir() as tmp_dir_path:
        with open(path_join(tmp_dir_path, 'data.dat'), 'wb') as data_file:
            # Write the bh_tsne header
            data_file.write(pack('iiddi', sample_count, sample_dim, theta, perplexity, no_dims))
            # Then write the data
            for sample in samples:
                data_file.write(pack(f'{len(sample)}d', *sample))
            # Write random seed if specified
            if randseed != EMPTY_SEED:
                data_file.write(pack('i', randseed))

        with open(devnull, 'w') as dev_null:
            bh_tsne_p = Popen([abspath(BH_TSNE_BIN_PATH)], cwd=tmp_dir_path, close_fds=True)
            bh_tsne_p.wait()
            assert not bh_tsne_p.returncode, (
                'ERROR: Call to bh_tsne exited with a non-zero return code exit status, please '
                f'{"enable verbose mode and " if not verbose else ""}refer to the bh_tsne output for further details'
            )

        with open(path_join(tmp_dir_path, 'result.dat'), 'rb') as output_file:
            result_samples, result_dims = _read_unpack('ii', output_file)
            results = [_read_unpack(f'{result_dims}d', output_file) for _ in range(result_samples)]
            results = [(_read_unpack('i', output_file), e) for e in results]
            results.sort()
            for _, result in results:
                yield result

def main(args):
    argp = _argparse().parse_args(args[1:])

    data = []
    try:
        for sample_line in argp.input:
            sample_data = list(map(float, sample_line.rstrip().split('\t')))
            if 'dims' in locals() and len(sample_data) != dims:
                raise ValueError(f"Input line has dimensionality {len(sample_data)} but expected {dims}")
            dims = len(sample_data)
            data.append(sample_data)
    except ValueError as e:
        print(f"Error processing input data: {e}", file=stderr)
        return 1

    for result in bh_tsne(np.array(data), no_dims=argp.no_dims, perplexity=argp.perplexity, theta=argp.theta, 
                          randseed=argp.randseed, verbose=argp.verbose, initial_dims=argp.initial_dims):
        argp.output.write("\t".join(map(str, result)) + "\n")

    return 0

if __name__ == '__main__':
    from sys import argv
    exit(main(argv))
