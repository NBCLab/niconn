from __future__ import division

from nipype.interfaces import fsl as fsl         # fsl
from nipype.interfaces import utility as util     # utility
from nipype.pipeline import engine as pe          # pypeline engine
import nipype.interfaces.io as nio
import os
import os.path as op
import shutil
from glob import glob
import numpy as np


def pickfirst(files):
    if isinstance(files, list):
        return files[0]
    else:
        return files


def calcres(volume, resels):
    resels = int(volume/resels)
    return resels


def rs_preprocess(in_file, fwhm, work_dir, output_dir):

    from nipype.workflows.fmri.fsl.preprocess import create_susan_smooth
    from nipype.interfaces.fsl.model import GLM
    from nipype.interfaces import fsl as fsl


    # define nodes and workflows
    rs_preproc_workflow = pe.Workflow(name="rs_preproc_workflow")
    rs_preproc_workflow.base_dir = work_dir

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func', 'fwhm']), name='inputspec')
    inputnode.inputs.func = in_file
    inputnode.inputs.fwhm = fwhm

    #make a brain mask
    immask = pe.Node(interface=fsl.ImageMaths(op_string = '-abs -bin -Tmin'), name='immask')
    rs_preproc_workflow.connect(inputnode, 'func', immask, 'in_file')

    #calculate mean image from smoothed data, for adding back after GSR
    immean = pe.Node(interface=fsl.ImageMaths(op_string = '-Tmean'), name='immean')
    rs_preproc_workflow.connect(inputnode, 'func', immean, 'in_file')

    #get time-series for GSR
    meants = pe.Node(interface=fsl.utils.ImageMeants(), name='meants')
    rs_preproc_workflow.connect(inputnode, 'func', meants, 'in_file')
    rs_preproc_workflow.connect(immask, 'out_file', meants, 'mask')

    #removing global signal
    glm = pe.Node(interface=GLM(), name='glm')
    glm.inputs.out_res_name = op.join(work_dir, 'res4d.nii.gz')
    rs_preproc_workflow.connect(inputnode, 'func', glm, 'in_file')
    rs_preproc_workflow.connect(immask, 'out_file', glm, 'mask')
    rs_preproc_workflow.connect(meants, 'out_file', glm, 'design')

    #add mean back to GSR'ed image
    maths = pe.Node(interface=fsl.maths.BinaryMaths(operation = 'add'), name='maths')
    rs_preproc_workflow.connect(glm, 'out_res', maths, 'in_file')
    rs_preproc_workflow.connect(immean, 'out_file', maths, 'operand_file')

    #do the smoothing
    smooth = create_susan_smooth()
    rs_preproc_workflow.connect(maths, 'out_file', smooth, 'inputnode.in_files')
    rs_preproc_workflow.connect(inputnode, 'fwhm', smooth, 'inputnode.fwhm')
    rs_preproc_workflow.connect(immask, 'out_file', smooth, 'inputnode.mask_file')

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    rs_preproc_workflow.connect(maths, 'out_file', datasink, 'gsr')
    rs_preproc_workflow.connect(immask, 'out_file', datasink, 'mask')
    rs_preproc_workflow.connect(smooth.get_node('smooth'), ('smoothed_file', pickfirst), datasink, 'gsr_smooth')

    rs_preproc_workflow.run()

    #copy data to directory
    gsr_fn = glob(op.join(work_dir, 'gsr', '*.nii.gz'))[0]
    mask_fn = glob(op.join(work_dir, 'mask', '*.nii.gz'))[0]
    gsr_smooth_fn = glob(op.join(work_dir, 'gsr_smooth', '*', '*.nii.gz'))[0]
    gsr_fn2 = op.join(output_dir, '{0}.nii.gz'.format(op.basename(in_file).split('.')[0]))
    mask_fn2 = op.join(output_dir, '{0}_mask.nii.gz'.format(op.basename(in_file).split('.')[0]))
    gsr_smooth_fn2 = op.join(output_dir, '{0}_smooth.nii.gz'.format(op.basename(in_file).split('.')[0]))

    shutil.copyfile(gsr_fn, gsr_fn2)
    shutil.copyfile(mask_fn, mask_fn2)
    shutil.copyfile(gsr_smooth_fn, gsr_smooth_fn2)

    shutil.rmtree(work_dir)


def rs_firstlevel(smooth_fn, roi_mask, output_dir):

    from nipype.interfaces import fsl as fsl
    import nibabel as nib
    import numpy as np


    mask_fn = "_".join(op.basename(smooth_fn).split('.')[0].split('_')[:-1])

    mask_img = nib.load(op.join(op.dirname(smooth_fn), '{prefix}_mask.nii.gz'.format(prefix=mask_fn)))
    mask_inds = np.nonzero(mask_img.get_fdata())

    smooth_img_datarray = nib.load(smooth_fn).get_fdata()[mask_inds]
    print(np.shape(smooth_img_datarray))

    colsum_smooth_data_array = np.sum(smooth_img_datarray, axis=1)
    colsum_smooth_data_array_square = np.sum(np.square(smooth_img_datarray), axis=1)
    square_colsum_smooth_data_array = np.square(colsum_smooth_data_array)

    for tmp_roi in roi_mask:

        #come back to this line
        tmp_output_dir = op.join(output_dir, 'derivatives', ppt, op.basename(smooth_fn).split('.')[0], tmp_roi.split('.')[0])
        if not op.isfile(op.join(tmp_output_dir, 'r.nii.gz')):
            roiimg = nib.load(tmp_roi)
            roiimg_inds = np.nonzero(roiimg.get_fdata())

            shared_inds_locs = [np.where(np.all(np.transpose(mask_inds) == x, axis=1))[0] for x in np.transpose(roiimg_inds)]
            shared_inds_locs = [loc for tup in shared_inds_locs for loc in tup]

            mean_ts = np.mean(smooth_img_datarray[shared_inds_locs,:], axis=0)

            corrdata = np.zeros(mask_img.shape)

            colsum_mean_ts = np.sum(mean_ts)
            colsum_mean_ts_square = np.sum(np.square(mean_ts))
            numerator = len(mean_ts)*np.sum(mean_ts * smooth_img_datarray, axis=1) - (colsum_mean_ts * colsum_smooth_data_array)
            denom = np.sqrt((len(mean_ts)*colsum_mean_ts_square - np.square(colsum_mean_ts)) * (len(mean_ts)*colsum_smooth_data_array_square - square_colsum_smooth_data_array))
            corrcoef = numerator/denom

            corrdata[mask_inds] = corrcoef

            corrimg = nib.Nifti1Image(corrdata, mask_img.affine)
            nib.save(corrimg, op.join(tmp_output_dir, 'r.nii.gz'))


def rs_secondlevel(r, output_dir):

    from nipype.interfaces import fsl as fsl
    import nibabel as nib
    import numpy as np


    img = np.zeros((np.prod(nib.load(r[0]).shape), len(r)))
    for i, rimg in enumerate(r):
        tmpimg = nib.load(rimg).get_fdata()
        img[:,i] = tmpimg.flatten()
    imgmin = np.abs(img).min(axis=1)
    imginds = np.nonzero(imgmin)[0]
    imgmean = np.mean(img[imginds,:], axis=1)
    imgmean[imgmean == 1] = 1 - np.finfo(float).eps
    imgout = np.zeros(nib.load(r[0]).shape)
    imgout[np.unravel_index(imginds, nib.load(r[0]).shape)] = np.arctanh(imgmean)
    nib.save(nib.Nifti1Image(imgout, nib.load(r[0]).affine), op.join(output_dir, 'z.nii.gz'))


def rs_grouplevel(z, prefix, output_dir, work_dir):

    from nipype.interfaces.fsl.model import Randomise
    from nipype.interfaces import fsl as fsl


    merger = fsl.utils.Merge()
    merger.inputs.in_files = z
    merger.inputs.dimension = 't'
    merger.inputs.merged_file = op.join(work_dir, 'z_merge.nii.gz')
    merger.run()

    immaths = fsl.utils.ImageMaths()
    immaths.inputs.in_file = op.join(work_dir, 'z_merge.nii.gz')
    immaths.inputs.op_string = "-abs -bin -Tmin"
    immaths.inputs.out_file = op.join(work_dir, 'mask.nii.gz')
    immaths.run()

    masker = fsl.maths.ApplyMask()
    masker.inputs.in_file = op.join(work_dir, 'z_merge.nii.gz')
    masker.inputs.mask_file = op.join(work_dir, 'mask.nii.gz')
    masker.inputs.out_file = op.join(work_dir, 'z_merge.nii.gz')
    masker.run()

    meaner = fsl.utils.ImageMaths()
    meaner.inputs.in_file = op.join(work_dir, 'z_merge.nii.gz')
    meaner.inputs.op_string = "-Tmean"
    meaner.inputs.out_file = op.join(work_dir, 'z.nii.gz')
    meaner.run()
    shutil.copyfile(op.join(work_dir, 'z.nii.gz'), op.join(output_dir, '{prefix}_z.nii.gz'.format(prefix=prefix)))

    randomise = Randomise()
    randomise.inputs.in_file = op.join(work_dir, 'z_merge.nii.gz')
    randomise.inputs.mask = op.join(work_dir, 'mask.nii.gz')
    randomise.inputs.one_sample_group_mean = True
    randomise.inputs.raw_stats_imgs = True
    randomise.inputs.vox_p_values = True
    randomise.inputs.base_name = 'randomise'
    os.chdir(work_dir)
    randomise.run()

    thresher = fsl.utils.ImageMaths()
    thresher.inputs.in_file = op.join(work_dir, 'randomise_vox_corrp_tstat1.nii.gz')
    thresher.inputs.op_string = "-thr 0.999"
    thresher.inputs.out_file = op.join(work_dir, 'randomise_vox_corrp_tstat1_thr001.nii.gz')
    thresher.run()

    masker = fsl.maths.ApplyMask()
    masker.inputs.in_file = op.join(work_dir, 'randomise_tstat1.nii.gz')
    masker.inputs.mask_file = op.join(work_dir, 'randomise_vox_corrp_tstat1_thr001.nii.gz')
    masker.inputs.out_file = op.join(work_dir, 'randomise_tstat1_thr001.nii.gz')
    masker.run()

    #copy data to directory vox_corrp_tstat1.nii.gz
    shutil.copyfile(op.join(work_dir, 'randomise_tstat1.nii.gz'), op.join(output_dir, '{prefix}_tstat1.nii.gz'.format(prefix=prefix)))
    shutil.copyfile(op.join(work_dir, 'randomise_vox_corrp_tstat1.nii.gz'), op.join(output_dir, '{prefix}_vox_corrp.nii.gz'.format(prefix=prefix)))
    shutil.copyfile(op.join(work_dir, 'randomise_tstat1_thr001.nii.gz'), op.join(output_dir, '{prefix}_tstat1_thr001.nii.gz'.format(prefix=prefix)))

    shutil.rmtree(work_dir)


def rs_workflow(coords=None, mask=None, rs_data_dir=None, work_dir=None):

    if coords is not None:
        from utils import make_sphere

        coords_str = '{x}_{y}_{z}'.format(x=str(coords[0]), y=str(coords[1]), z=str(coords[2]))
        roi_mask_fn = [op.join(work_dir, '{coords_str}.nii.gz'.format(coords_str=coords_str))]

        make_sphere(coords[0], coords[1], coords[2], work_dir)

    elif mask is not None:
        roi_mask_fn = mask

    from nipype.interfaces.base import Bunch
    import pandas as pd

    #get participants
    ppt_df = pd.read_csv(op.join(rs_data_dir, 'hcp1200_participants-150.tsv'), sep='/t')
    for ppt in ppt_df['participant_id']:
        ppt = str(ppt)
        nii_files = sorted(glob(op.join(rs_data_dir, ppt, 'MNINonLinear', 'Results', 'rfMRI_REST*', 'rfMRI_REST*_hp2000_clean.nii.gz')))
        nii_files = [x for x in nii_files if '7T' not in x]
        for nii_fn in nii_files:

            #check to see if smoothed data exists
            tmp_output_dir = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt)
            smooth_fn = op.join(tmp_output_dir, '{0}_smooth.nii.gz'.format(op.basename(nii_fn).split('.')[0]))
            if not op.isfile(smooth_fn):
                if not op.isdir(tmp_output_dir):
                    os.makedirs(tmp_output_dir)
                nii_work_dir = op.join(work_dir, 'rsfc', ppt, op.basename(nii_fn).split('.')[0])
                rs_preprocess(nii_fn, 4, nii_work_dir, tmp_output_dir)

            #run analysis
            tmp_output_dir = op.join(rs_data_dir, 'derivatives', ppt, op.basename(nii_fn).split('.')[0])
            os.makedirs(tmp_output_dir, exist_ok=True)
            rs_firstlevel(nii_fn, smooth_fn, roi_mask_fn, tmp_output_dir)

        for tmp_roi in roi_mask_fn:
            output_dir = op.join(rs_data_dir, 'derivatives', ppt, op.basename(tmp_roi).split('.')[0])
            if not op.isfile(op.join(output_dir, 'z.nii.gz')):
                r = [sorted(glob(op.join(rs_data_dir, 'derivatives', ppt, '*', op.basename(tmp_roi).split('.')[0], 'r.nii.gz')))]
                rs_secondlevel(r, output_dir)

    for tmp_roi in roi_mask_fn:
        z = sorted(glob(op.join(rs_data_dir, 'derivatives', '*', op.basename(tmp_roi).split('.')[0], 'z.nii.gz')))

        output_dir = op.join(rs_data_dir, 'derivatives', op.basename(tmp_roi).split('.')[0])
        os.makedirs(output_dir, exist_ok=True)

        nii_work_dir = op.join(work_dir, 'rsfc', op.basename(tmp_roi).split('.')[0])
        os.makedirs(nii_work_dir, exist_ok=True)

        rs_grouplevel(z, coords_str, output_dir, nii_work_dir)
