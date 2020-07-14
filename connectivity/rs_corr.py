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

    #do the smoothing
    smooth = create_susan_smooth()
    rs_preproc_workflow.connect(inputnode, 'func', smooth, 'inputnode.in_files')
    rs_preproc_workflow.connect(inputnode, 'fwhm', smooth, 'inputnode.fwhm')
    rs_preproc_workflow.connect(immask, 'out_file', smooth, 'inputnode.mask_file')

    #calculate mean image from smoothed data, for adding back after GSR
    immean = pe.Node(interface=fsl.ImageMaths(op_string = '-Tmean'), name='immean')
    rs_preproc_workflow.connect(smooth.get_node('smooth'), ('smoothed_file', pickfirst), immean, 'in_file')

    #get time-series for GSR
    meants = pe.Node(interface=fsl.utils.ImageMeants(), name='meants')
    rs_preproc_workflow.connect(inputnode, 'func', meants, 'in_file')
    rs_preproc_workflow.connect(immask, 'out_file', meants, 'mask')

    #removing global signal
    glm = pe.Node(interface=GLM(), name='glm')
    glm.inputs.out_res_name = op.join(work_dir, 'res4d.nii.gz')
    rs_preproc_workflow.connect(smooth.get_node('smooth'), ('smoothed_file', pickfirst), glm, 'in_file')
    rs_preproc_workflow.connect(immask, 'out_file', glm, 'mask')
    rs_preproc_workflow.connect(meants, 'out_file', glm, 'design')

    #add mean back to GSR'ed image
    maths = pe.Node(interface=fsl.maths.BinaryMaths(operation = 'add'), name='maths')
    rs_preproc_workflow.connect(glm, 'out_res', maths, 'in_file')
    rs_preproc_workflow.connect(immean, 'out_file', maths, 'operand_file')

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    #rs_preproc_workflow.connect(featproc, 'meanscale.out_file', datasink, 'gms')
    rs_preproc_workflow.connect(maths, 'out_file', datasink, 'gsr')
    rs_preproc_workflow.connect(immask, 'out_file', datasink, 'mask')

    rs_preproc_workflow.run()

    #copy data to directory
    #gms_fn = glob(op.join(work_dir, 'gms', '_meanscale0', '*_gms.nii.gz'))[0]
    gsr_fn = glob(op.join(work_dir, 'gsr', '*.nii.gz'))[0]
    mask_fn = glob(op.join(work_dir, 'mask', '*.nii.gz'))[0]
    gsr_fn2 = op.join(output_dir, '{0}_smooth.nii.gz'.format(op.basename(in_file).split('.')[0]))
    mask_fn2 = op.join(output_dir, '{0}_mask.nii.gz'.format(op.basename(in_file).split('.')[0]))

    shutil.copyfile(gsr_fn, gsr_fn2)
    shutil.copyfile(mask_fn, mask_fn2)

    shutil.rmtree(work_dir)


def rs_firstlevel(smooth_fn, roi_mask, output_dir, work_dir):

    from nipype.interfaces import fsl as fsl
    import nibabel as nib
    import numpy as np


    meants = fsl.utils.ImageMeants()
    meants.inputs.in_file = smooth_fn
    meants.inputs.mask = roi_mask
    meants.inputs.out_file = op.join(work_dir, '{0}_{1}.txt'.format(op.basename(smooth_fn).split('.')[0], op.basename(roi_mask).split('.')[0]))
    meants.run()

    mask_fn = "_".join(op.basename(smooth_fn).split('.')[0].split('_')[:-1])

    roi_ts = np.atleast_2d(np.loadtxt(op.join(work_dir, '{0}_{1}.txt'.format(op.basename(smooth_fn).split('.')[0], op.basename(roi_mask).split('.')[0]))))

    mask_img = nib.load(op.join(op.dirname(smooth_fn), '{prefix}_mask.nii.gz'.format(prefix=mask_fn)))
    mask_inds = np.nonzero(mask_img.get_fdata())

    smooth_img = nib.load(smooth_fn)
    smooth_img_datarray = smooth_img.get_fdata()[mask_inds]

    corrdata = np.zeros(mask_img.shape)
    corrcoef = np.zeros(np.shape(smooth_img_datarray)[0])

    for i, ts in enumerate(smooth_img_datarray):
        corrcoef[i] = np.corrcoef(ts, roi_ts)[0][1]

    corrdata[mask_inds] = corrcoef

    corrimg = nib.Nifti1Image(corrdata, mask_img.affine)
    nib.save(corrimg, op.join(output_dir, 'r.nii.gz'))

    shutil.rmtree(work_dir)


def rs_secondlevel(z, output_dir, work_dir):

    from nipype.interfaces import fsl as fsl
    import nibabel as nib
    import numpy as np


    merger = fsl.utils.Merge()
    merger.inputs.in_files = z
    merger.inputs.dimension = 't'
    merger.inputs.merged_file = op.join(work_dir, 'r_merge.nii.gz')
    merger.run()

    immaths = fsl.utils.ImageMaths()
    immaths.inputs.in_file = op.join(work_dir, 'r_merge.nii.gz')
    immaths.inputs.op_string = "-abs -bin -Tmin"
    immaths.inputs.out_file = op.join(work_dir, 'mask.nii.gz')
    immaths.run()

    masker = fsl.maths.ApplyMask()
    masker.inputs.in_file = op.join(work_dir, 'r_merge.nii.gz')
    masker.inputs.mask_file = op.join(work_dir, 'mask.nii.gz')
    masker.inputs.out_file = op.join(work_dir, 'r_merge.nii.gz')
    masker.run()

    meaner = fsl.utils.ImageMaths()
    meaner.inputs.in_file = op.join(work_dir, 'r_merge.nii.gz')
    meaner.inputs.op_string = "-Tmean"
    meaner.inputs.out_file = op.join(work_dir, 'r.nii.gz')
    meaner.run()

    rimg = nib.load(op.join(work_dir, 'r.nii.gz'))
    maskimg = nib.load(op.join(work_dir, 'mask.nii.gz'))

    mask_inds = np.nonzero(maskimg.get_fdata())

    rdata = rimg.get_fdata()[mask_inds]
    rdata[rdata == 1] = 1 - np.finfo(float).eps

    zdata = np.zeros(maskimg.shape)
    zdata[mask_inds] = np.arctanh(rdata)

    zimg = nib.Nifti1Image(zdata, maskimg.affine)
    nib.save(zimg, op.join(output_dir, 'z.nii.gz'))

    #copy data to directory
    shutil.copy(op.join(work_dir, 'r.nii.gz'), output_dir)
    shutil.rmtree(work_dir)


def rs_grouplevel(copes, varcopes, dofs, output_dir, work_dir):

    from nipype.interfaces.fsl.model import MultipleRegressDesign
    from nipype.interfaces.utility import Function
    from nipype.interfaces.fsl.model import FLAMEO
    from nipype.interfaces.fsl.model import SmoothEstimate
    from connectivity.interfaces import Cluster
    from nipype.interfaces.fsl.utils import Merge
    from nipype.interfaces.fsl import Info
    from connectivity.interfaces import PtoZ

    grplevelworkflow = pe.Workflow(name="grplevelworkflow")
    grplevelworkflow.base_dir = work_dir

    merger = Merge()
    merger.inputs.dimension = 't'
    merger.inputs.in_files = copes
    merger.inputs.merged_file = op.join(work_dir, 'cope.nii.gz')
    merger.run()

    merger.inputs.in_files = varcopes
    merger.inputs.merged_file = op.join(work_dir, 'varcope.nii.gz')
    merger.run()

    merger.inputs.in_files = dofs
    merger.inputs.merged_file = op.join(work_dir, 'dof.nii.gz')
    merger.run()

    model = pe.Node(interface=MultipleRegressDesign(), name='model')
    model.inputs.contrasts = [['mean', 'T', ['roi'], [1]]]
    model.inputs.regressors = dict(roi=np.ones(len(copes)).tolist())

    flameo = pe.Node(interface=FLAMEO(), name='flameo')
    flameo.inputs.cope_file = op.join(work_dir, 'cope.nii.gz')
    flameo.inputs.var_cope_file = op.join(work_dir, 'varcope.nii.gz')
    flameo.inputs.dof_var_cope_file = op.join(work_dir, 'dof.nii.gz')
    flameo.inputs.run_mode = 'flame1'
    flameo.inputs.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

    grplevelworkflow.connect(model, 'design_con', flameo, 't_con_file')
    grplevelworkflow.connect(model, 'design_grp', flameo, 'cov_split_file')
    grplevelworkflow.connect(model, 'design_mat', flameo, 'design_file')

    smoothest = pe.Node(SmoothEstimate(), name='smooth_estimate')
    grplevelworkflow.connect(flameo, 'zstats', smoothest, 'residual_fit_file')
    smoothest.inputs.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')
    smoothest.inputs.dof = len(dofs)-1

    cluster = pe.Node(Cluster(), name='cluster')
    ptoz = pe.Node(PtoZ(), name='ptoz')
    ptoz.inputs.pvalue = 0.001
    calculate_resels = pe.Node(Function(input_names=["volume", "resels"], output_names=["resels"], function=calcres), name="calcres")
    grplevelworkflow.connect(smoothest, 'resels', cluster, 'resels')
    grplevelworkflow.connect(smoothest, 'resels', calculate_resels, 'resels')
    grplevelworkflow.connect(smoothest, 'volume', calculate_resels, 'volume')
    grplevelworkflow.connect(calculate_resels, 'resels', ptoz, 'resels')
    grplevelworkflow.connect(ptoz, 'zstat', cluster, 'threshold')
    cluster.inputs.connectivity = 26
    cluster.inputs.out_threshold_file = True
    cluster.inputs.out_index_file = True
    cluster.inputs.out_localmax_txt_file = True
    cluster.inputs.voxthresh = True

    grplevelworkflow.connect(flameo, 'zstats', cluster, 'in_file')

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    grplevelworkflow.connect(flameo, 'zstats', datasink, 'z')
    grplevelworkflow.connect(cluster, 'threshold_file', datasink, 'z_thresh')

    grplevelworkflow.run()

    shutil.rmtree(op.join(work_dir, 'grplevelworkflow'))
    #copy data to directory
    shutil.copyfile(op.join(work_dir, 'z', 'zstat1.nii.gz'), op.join(output_dir, 'z.nii.gz'))
    shutil.copyfile(op.join(work_dir, 'z_thresh', 'zstat1_threshold.nii.gz'), op.join(output_dir, 'z_level-voxel_corr-FWE.nii.gz'))

    shutil.rmtree(work_dir)


def rs_workflow(x=None, y=None, z=None, rs_data_dir=None, work_dir=None):

    from utils import make_sphere

    coords_str = '{x}_{y}_{z}'.format(x=str(x), y=str(y), z=str(z))
    roi_mask_fn = op.join(work_dir, '{coords_str}.nii.gz'.format(coords_str=coords_str))

    make_sphere(x, y, z, work_dir)

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
            tmp_output_dir = op.join(rs_data_dir, 'derivatives', coords_str, ppt, op.basename(nii_fn).split('.')[0])
            if not op.isfile(op.join(tmp_output_dir, 'z.nii.gz')):
                if not op.isdir(tmp_output_dir):
                    os.makedirs(tmp_output_dir)
                nii_work_dir = op.join(work_dir, 'rsfc', coords_str, ppt, op.basename(nii_fn).split('.')[0])
                if not op.isdir(nii_work_dir):
                    os.makedirs(nii_work_dir)
                rs_firstlevel(nii_fn, smooth_fn, roi_mask_fn, tmp_output_dir, nii_work_dir)

        output_dir = op.join(rs_data_dir, 'derivatives', coords_str, ppt)
        if not op.isfile(op.join(output_dir, 'z.nii.gz')):

            if len(nii_files)>1:

                z = [sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, ppt, '*', 'z.nii.gz')))]
                output_dir = op.join(rs_data_dir, 'derivatives', coords_str, ppt)
                nii_work_dir = op.join(work_dir, 'rsfc', coords_str, ppt)
                rs_secondlevel(z, output_dir, nii_work_dir)

            else:

                stat_files = sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, ppt, '*', 'z.nii.gz')))
                shutil.copy(stat_files[0], output_dir)

    z = sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, '*', 'z.nii.gz')))

    output_dir = op.join(rs_data_dir, 'derivatives', coords_str)
    if not op.isdir(output_dir):
        os.makedirs(output_dir)

    nii_work_dir = op.join(work_dir, 'rsfc', coords_str)
    if not op.isdir(nii_work_dir):
        os.makedirs(nii_work_dir)

    rs_grouplevel(z, output_dir, nii_work_dir)
