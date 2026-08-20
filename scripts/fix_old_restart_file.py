#!/usr/bin/env python3

# Fix an (any) KHARMA restart file so that KHARMA can restart from it
# this works around a bug in Parthenon w.r.t. mesh sizes

import sys
import shutil
import numpy as np
import h5py

shutil.copyfile(sys.argv[1], sys.argv[2])
outf = h5py.File(sys.argv[2], "r+")

# Parthenon records the full size here, so we amend it to exclude ghost zones
outf['Info'].attrs.modify('MeshBlockSize',
  np.maximum(outf['Info'].attrs['MeshBlockSize'][()] - 2*outf['Info'].attrs['IncludesGhost'][()]*outf['Info'].attrs['NGhost'][()],
             np.ones_like(outf['Info'].attrs['MeshBlockSize'][()])))

outf.close()
