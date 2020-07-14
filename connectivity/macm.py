import os
import os.path as op
from nimare.meta.cbma import ALE
from nimare.dataset import Dataset
from nimare.correct import FWECorrector
import pickle

def macm(prefix=None, mask=None, output_dir=None, ns_data_dir=None):

    if mask is None or not op.isfile(mask):
        raise Exception('A valid mask is required for input!')

    if ns_data_dir is None:
        raise Exception('A valid directory is required for downloading Neurosynth data!')

    if prefix is None:
        prefix = op.basename(mask).split('.')[0]

    if output_dir is None:
        output_dir = op.dirname(op.abspath(mask))

    dataset_file = op.join(ns_data_dir, 'neurosynth_dataset.pkl.gz')

    # download neurosynth dataset if necessary
    if not op.isfile(dataset_file):
        from datasets.neurosynth import neurosynth_download
        neurosynth_download(ns_data_dir)

    dset = Dataset.load(dataset_file)
    mask_ids = dset.get_studies_by_mask(mask)
    maskdset = dset.slice(mask_ids)
    nonmask_ids = sorted(list(set(dset.ids) - set(mask_ids)))
    nonmaskdset = dset.slice(nonmask_ids)

    ale = ALE(kernel__fwhm=15)
    ale.fit(maskdset)

    corr = FWECorrector(method='permutation', n_iters=5000, n_cores=-1, voxel_thresh=0.001)
    cres = corr.transform(ale.results)
    cres.save_maps(output_dir=output_dir, prefix=prefix)


def macm_workflow(x=None, y=None, z=None, ns_data_dir=None, output_dir=None):

    from .utils import make_sphere

    coords_str = '{x}_{y}_{z}'.format(x=str(x), y=str(y), z=str(z))
    roi_mask_fn = op.join(output_dir, '{coords_str}.nii.gz'.format(coords_str=coords_str))

    make_sphere(x, y, z, output_dir)

    # download neurosynth dataset if necessary
    dataset_file = op.join(ns_data_dir, 'neurosynth_dataset.pkl.gz')
    if not op.isfile(dataset_file):
        from datasets.neurosynth import neurosynth_download
        neurosynth_download(ns_data_dir=ns_data_dir)

    macm(prefix=None, mask=roi_mask_fn, output_dir=output_dir, ns_data_dir=ns_data_dir)
