"""
Unit cell stacker
=================

This code can be used for stacking the unit cells of different content and
dimension to a single cell. This is intended to be helpful for constructing
complex super cell structures.

The unit cells should be given in Gaussian format, where the lattice vectors are
input in the same way as atomic coordinates, just the element symbol is set to
'Tv'. The way the cells are going to be stacked should be given in a YAML file,
whose name is given in the command line arguments. Also it needs to be noted
that the current code just works for cubic unit cells.

The YAML file contains a large nested list. The first level are entries
corresponding to each layer in the Z direction of the super cell, given in the
order from ground up. For each layer, a list of rows are given from small Y
coordinate value to larger. And within each row list, a list of building blocks
should be given from small x value to larger. So in summary, the nested
structure goes as

1. Layer (Z direction)
2. Row (Y direction)
3. Block (X direction).

And for all three levels, the order is consistently from smaller to larger
coordinate values.

Each block is going to be given as a dictionary, with the following keys,

unit
  The base name of the file containing the unit cell of this building block. The
  actual file name should end with ``.gjf``. If the ``prefix`` parameter is set
  in the parameters, then the file name is going to be prepended with this, or
  it will be tried to be found in the current working directory.

repetition
  A list giving the repetition of the unit cell. It can be omitted for no
  repetition, i.e. ``[1, 1, 1]``. The entries can be integers or string. If
  string is given, then the actual integral number is going to be resolved from
  the second document of the YAML file. If no symbol is used, the second
  document can just be omitted.

The file name of the output file can be given in the command line argument
``-o``, which has got the default value of the standard output. Also some
additional parameters can be given by a file given by the parameter ``-p``,
which is also in YAML format and will be combined with the second document of
the main input.

"""

from __future__ import print_function

import re
import collections
import itertools
import argparse
import sys

import numpy as np
import yaml


#
# The Gaussian file reader
# ------------------------
#

def read_gaussian(file_name):

    """Reads a Gaussian input file

    Returns a list of atoms and the three lattice vectors in a pair.

    """

    with open(file_name, 'r') as inp:

        # form the patterns
        symbol_pattern = r'^\s*(?P<symbol>\d{1,2}|[A-Z][a-z]?\d*)'
        float_pattern = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
        coord_names = ['coord%d' % i for i in xrange(0, 3)]
        coord_patterns = ['(?P<%s>%s)' % (i, float_pattern)
                          for i in coord_names]
        atm_pattern = re.compile('\\s+'.join([symbol_pattern, ] + coord_patterns))

        # read the coordinates
        coords = []
        for line in inp:
            match_res = atm_pattern.match(line)
            if match_res is not None:
                coords.append(
                    (
                        match_res.group('symbol'),
                        np.array(match_res.group(*coord_names),
                                 dtype=np.float64)
                    )
                )
            continue

        # filter the atoms and the lattice vectors
        atms = [i for i in coords if i[0] != 'Tv']
        latt_vecs = [i[1] for i in coords if i[0] == 'Tv']
        if len(latt_vecs) != 3:
            raise ValueError('Wrong number of lattice vectors')

    return atms, latt_vecs


#
# Generate the stacking data structure
# ------------------------------------
#
# The YAML input file is tried to be read and parsed. In this process, the unit
# cells are also read, and the symbolic repetition numbers are also resolved
# based on the values that are given in the input file. Then the processed
# blocks are put in the same nested list data structure as in the input file.
# And this nested list structure is going to be termed a ``stacking``.
#

Block = collections.namedtuple('Block',
                               [
                                   'atms',
                                   'latt_dims',
                                   'repetition'
                               ])

# Note that since the code is just designed to work for square unit cells, the
# lattice vectors, the lattice dimensions of the x, y, and z are stored instead
# of the full lattice vectors here.


def gen_stacking(main_inp, additional_inp):

    """Generates a stacking based on the YAML input file name

    It will just return the same data structure as the input, with the unit
    cell and the symbolic repetition number resolved.

    :param main_inp: The primary input file.
    :param additional_inp: The input file for additional parameters, None for no
        additional parameters

    """

    yaml_docs = list(yaml.load_all(main_inp))

    raw_stacking = yaml_docs[0]
    if len(yaml_docs) < 2:
        params = {}
    else:
        params = yaml_docs[1]

    if additional_inp is not None:
        additional_params = yaml.load(additional_inp)
        params.update(additional_params)

    unit_cells = {}

    def process_raw_dict(raw_dict):

        """The closure for process a raw dictionary read for a block"""

        # resolve the unit cell
        unit_base_name = raw_dict['unit']
        unit_prefix = params.get('prefix', '')
        unit_file_name = unit_prefix + unit_base_name + '.gjf'
        if unit_base_name in unit_cells:
            atms, latt_vecs = unit_cells[unit_base_name]
        else:
            atms, latt_vecs = read_gaussian(unit_file_name)
            unit_cells[unit_base_name] = (atms, latt_vecs)
        latt_dims = [latt_vecs[i][i] for i in xrange(0, 3)]

        # resolve the repetition
        dummy_glob = {}
        if 'repetition' in raw_dict:
            raw_repetition = raw_dict['repetition']
            try:
                repetition = [
                    i if type(i) == int else eval(i, dummy_glob, params)
                    for i in raw_repetition
                ]
            except IndexError:
                print('Symbolic repetition number cannot be resolved!',
                      file=sys.stderr)
                raise
        else:
            repetition = [1, 1, 1]

        return Block(atms=atms, latt_dims=latt_dims, repetition=repetition)

    # return the three level nested list
    return [
        [
            [
                process_raw_dict(block_i)
                for block_i in row_i
            ]
            for row_i in layer_i
        ]
        for layer_i in raw_stacking
    ]


#
# Perform the actual stacking
# ---------------------------
#
# After the stacking data structure has been generated, all the information
# that is needed for obtaining the stacked super cell is ready. This is
# achieved in the function :py:func`do_stacking`, which returns the atoms and
# the lattice vectors of the stacked structure.
#

def translate(atms, vec):

    """Translate a list of atoms

    :param atms: The list of atoms, given as a pair of the symbol and the
        coordinate in numpy array.
    :param vec: A numpy array giving the translation vector.

    """

    return [
        (i[0], i[1] + vec)
        for i in atms
    ]


def do_stacking(stacking):

    """Does the actual stacking

    The pair of atom list and the lattice vectors are going to be returned.

    """

    # This function is going to be written in a pure imperative style.

    cur_begin = [0.0 for i in xrange(0, 3)]
    cur_end = [0.0 for i in xrange(0, 3)]
    atms = []

    for layer_i in stacking:
        for row_i in layer_i:
            for block_i in row_i:

                # Add the atoms
                block_transl = np.array(cur_begin)
                for rep_i in itertools.product(
                        *[xrange(0, i) for i in block_i.repetition]
                        ):
                    rep_transl = np.array(
                        [i * j
                         for i, j in itertools.izip(block_i.latt_dims, rep_i)]
                        )
                    transl = block_transl + rep_transl
                    new_atms = translate(block_i.atms, transl)
                    atms.extend(new_atms)

                # Compute the end point and update the cur_end
                end_point = cur_begin + np.array(
                    [i * j for i, j in itertools.izip(block_i.repetition,
                                                      block_i.latt_dims)]
                    )
                cur_end = [
                    max(i, j) for i, j in itertools.izip(cur_end, end_point)
                ]

                # Update the beginning point for the next block
                cur_begin[0] = end_point[0]

                continue
                # end looping over blocks

            cur_begin[0] = 0.0
            cur_begin[1] = end_point[1]
            continue
            # end looping over rows

        cur_begin[0] = 0.0
        cur_begin[1] = 0.0
        cur_begin[2] = end_point[2]
        continue
        # end looping over layers

    latt_vecs = [np.zeros(3) for i in xrange(0, 3)]
    for i in xrange(0, 3):
        latt_vecs[i][i] = cur_end[i]
        continue

    return atms, latt_vecs


#
# Final printing
# --------------
#
# After we got the atoms and the lattice vectors, we need to print them out
#

def dump_coord(stream, atms, latt_vecs):

    """Dumps the atomic coordinates and the lattice vectors to the stream"""

    all_coords = atms + [
        ('Tv', i) for i in latt_vecs
    ]

    for i in all_coords:
        print(' %s  %f %f %f ' % ((i[0], ) + tuple(i[1])), file=stream)
        continue

    return None


#
# Driver function
# ---------------
#

def main():

    """The main driver function"""

    # parse the arguments
    parser = argparse.ArgumentParser(description='Stack unit cells')
    parser.add_argument('input', metavar='INPUT',
                        type=argparse.FileType('r'),
                        help='The YAML input file')
    parser.add_argument('-o', '--output', metavar='OUTPUT',
                        type=argparse.FileType(mode='w'), default=sys.stdout,
                        help='The output file name')
    parser.add_argument('-p', '--parameters', metavar='FILE',
                        type=argparse.FileType(mode='r'), default=None,
                        help='YAML file for additional parameters')
    args = parser.parse_args()

    # Read the input, generate the stacking
    stacking = gen_stacking(args.input, args.parameters)

    # perform the stacking
    atms, latt_vecs = do_stacking(stacking)

    # Dump the output
    dump_coord(args.output, atms, latt_vecs)

    return 0


# The sentinel
if __name__ == '__main__':
    main()
