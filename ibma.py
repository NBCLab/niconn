
def fishers_workflow(img_list=None, prefix=None, output_dir=None):

    from nimare.meta.esma import fishers
    from nilearn.datasets import load_mni152_brain_mask
    import nibabel as nib
    import numpy as np
    import os.path as op
    import statsmodels.stats.multitest as mc

    mask_img = load_mni152_brain_mask()
    mask_ind = np.nonzero(mask_img.get_data())

    img_stack = []
    for i, img_fn in enumerate(img_list):

        tmp_img = nib.load(img_fn)

        if i == 0:
            img_stack = tmp_img.get_data()[mask_ind]
        else:
            img_stack = np.vstack([img_stack, tmp_img.get_data()[mask_ind]])

    results = fishers(img_stack, two_sided=False)

    for tmp_key in results.keys():

        img_data = np.zeros(mask_img.shape)

        img_data[mask_ind] = results[tmp_key]
        img = nib.Nifti1Image(img_data, mask_img.affine)

        nib.save(img, op.join(output_dir, '{prefix}_{suffix}.nii.gz'.format(prefix=prefix, suffix=tmp_key)))

    _, p_corr = mc.fdrcorrection(results['p'], alpha=0.05, method='indep',
                                     is_sorted=False)

    img_data = np.zeros(mask_img.shape)

    img_data[mask_ind] = p_corr
    img = nib.Nifti1Image(img_data, mask_img.affine)

    nib.save(img, op.join(output_dir, '{prefix}_p_corr-fdr05.nii.gz'.format(prefix=prefix)))
