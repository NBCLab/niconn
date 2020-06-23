import os
import os.path as op
from neurosynth.base.dataset import download
from nimare.io import convert_neurosynth_to_dataset


def neurosynth_download(ns_data_dir=None):

    if ns_data_dir is None:
        raise Exception('A valid directory is required for downloading Neurosynth data!')

    dataset_file = op.join(ns_data_dir, 'neurosynth_dataset.pkl.gz')

    if not op.isdir(ns_data_dir):
        os.mkdir(ns_data_dir)

    download(ns_data_dir, unpack=True)
    ###############################################################################
    # Convert Neurosynth database to NiMARE dataset file
    # --------------------------------------------------
    dset = convert_neurosynth_to_dataset(
        op.join(ns_data_dir, 'database.txt'),
        op.join(ns_data_dir, 'features.txt'))
    dset.save(dataset_file)
