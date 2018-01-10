from __future__ import print_function, absolute_import

import os
import MDAnalysis as mda


def write_frame(a_universe, iframe, file_name):
    r"""Write a single trajectory frame to file.

    Format is guessed from the file's extension.

    Parameters
    ----------
    a_universe : :class:`~MDAnalysis.core.universe.Universe`
        Universe describing the simulation
    iframe : int
        Trajectory frame index (indexes begin with zero)
    file_name : str
        Name of the file to create
    """
    a_universe.trajectory[iframe]
    # Create directory if not existing
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    with mda.Writer(file_name) as writer:
        writer.write(a_universe)
