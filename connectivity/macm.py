import os
import os.path as op
from nimare.meta.cbma import ALE
from nimare.dataset import Dataset
from nimare.correct import FWECorrector
from neurosynth.base.dataset import download
from nimare.io import convert_neurosynth_to_dataset


def macm_workflow(ns_data_dir, output_dir, mask_fn):

    prefix = op.basename(mask_fn).split('.')[0]
    # download neurosynth dataset if necessary
    dataset_file = op.join(ns_data_dir, 'neurosynth_dataset.pkl.gz')

    if not op.isfile(dataset_file):
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

    dset = Dataset.load(dataset_file)
    mask_ids = dset.get_studies_by_mask(mask_fn)
    maskdset = dset.slice(mask_ids)
    nonmask_ids = sorted(list(set(dset.ids) - set(mask_ids)))
    nonmaskdset = dset.slice(nonmask_ids)

    ale = ALE(kernel__fwhm=15)
    ale.fit(maskdset)

    corr = FWECorrector(method='permutation', n_iters=10, n_cores=-1, voxel_thresh=0.001)
    cres = corr.transform(ale.results)
    cres.save_maps(output_dir=output_dir, prefix=prefix)
