import os
import pytest
import numpy as np
import MDAnalysis as mda
import shutil

from idpflex import utils
from idpflex.properties import SaxsProperty, SansProperty


def test_write_frame(trajectory_benchmark):
    utils.write_frame(trajectory_benchmark, 0, 'test.pdb')
    with open('test.pdb') as handle:
        content = handle.read()
        assert 'ATOM    535  OC2 TYR S  37      47.180  41.030' in content
    os.remove('test.pdb')


@pytest.mark.skipif(shutil.which('cryson') is None
                    or shutil.which('crysol') is None, reason='Needs cryson')
def test_generate_trajectory_profiles(saxs_benchmark, sans_benchmark):
    saxs_ref = np.loadtxt(saxs_benchmark['frame_profile'])
    sans_ref = np.loadtxt(sans_benchmark['frame_profile'])
    universe = mda.Universe(saxs_benchmark['crysol_pdb'],
                            saxs_benchmark['crysol_xtc'])
    protein = universe.select_atoms('protein')
    saxs_prof = utils.generate_trajectory_profiles(protein, range(4),
                                                   SaxsProperty)[3].profile
    sans_prof = utils.generate_trajectory_profiles(protein, range(4),
                                                   SansProperty)[3].profile
    np.testing.assert_array_almost_equal(saxs_ref, saxs_prof)
    np.testing.assert_array_almost_equal(sans_ref, sans_prof)


if __name__ == '__main__':
    pytest.main()
