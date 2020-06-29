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


def calcres(smoothest_input):
    resels = int(228453/smoothest_input)
    return resels


def rs_preprocess(in_file, fwhm, work_dir, output_dir):

    from nipype.workflows.fmri.fsl.preprocess import create_featreg_preproc

    rs_preproc_workflow = pe.Workflow(name="rs_preproc_workflow")
    rs_preproc_workflow.base_dir = work_dir

    featproc = create_featreg_preproc(name="featproc", highpass=False, whichvol='first')
    featproc.inputs.inputspec.func = in_file
    featproc.inputs.inputspec.fwhm = fwhm

    #remove motion correction nodes
    moco = featproc.get_node('realign')
    moplot = featproc.get_node('plot_motion')
    featproc.remove_nodes([moco, moplot])

    #remove connections dependent on motion correction
    featproc.disconnect(featproc.get_node('img2float'), 'out_file', featproc.get_node('motion_correct'), 'in_file')
    featproc.disconnect(featproc.get_node('extract_ref'), 'roi_file', featproc.get_node('motion_correct'), 'ref_file')
    featproc.disconnect(featproc.get_node('motion_correct'), ('mean_img', pickfirst), featproc.get_node('outputnode'), 'reference')
    featproc.disconnect(featproc.get_node('motion_correct'), 'par_file', featproc.get_node('outputnode'), 'motion_parameters')
    featproc.disconnect(featproc.get_node('motion_correct'), 'out_file', featproc.get_node('outputnode'), 'realigned_files')
    featproc.disconnect(featproc.get_node('motion_correct'), 'par_file', featproc.get_node('plot_motion'), 'in_file')
    featproc.disconnect(featproc.get_node('plot_motion'), 'out_file', featproc.get_node('outputnode'), 'motion_plots')
    featproc.disconnect(featproc.get_node('motion_correct'), ('out_file', pickfirst), featproc.get_node('meanfunc'), 'in_file')
    featproc.disconnect(featproc.get_node('motion_correct'), 'out_file', featproc.get_node('maskfunc'), 'in_file')
    featproc.disconnect(featproc.get_node('motion_correct'), 'out_file', featproc.get_node('medianval'), 'in_file')
    featproc.disconnect(featproc.get_node('motion_correct'), 'out_file', featproc.get_node('maskfunc2'), 'in_file')

    #add connections to fill in where motion correction files would have been entered
    featproc.connect(featproc.get_node('img2float'), ('out_file', pickfirst), featproc.get_node('meanfunc'), 'in_file')
    featproc.connect(featproc.get_node('img2float'), ('out_file', pickfirst), featproc.get_node('maskfunc'), 'in_file')
    featproc.connect(featproc.get_node('img2float'), ('out_file', pickfirst), featproc.get_node('medianval'), 'in_file')
    featproc.connect(featproc.get_node('img2float'), ('out_file', pickfirst), featproc.get_node('maskfunc2'), 'in_file')

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    rs_preproc_workflow.connect(featproc, 'meanscale.out_file', datasink, 'gms')
    rs_preproc_workflow.connect(featproc, 'dilatemask.out_file', datasink, 'mask')

    rs_preproc_workflow.run()

    #copy data to directory
    gms_fn = glob(op.join(work_dir, 'gms', '_meanscale0', '*_gms.nii.gz'))[0]
    mask_fn = glob(op.join(work_dir, 'mask', '_dilatemask0', '*_dil.nii.gz'))[0]
    gms_fn2 = op.join(output_dir, '{0}_smooth.nii.gz'.format(op.basename(in_file).split('.')[0]))
    mask_fn2 = op.join(output_dir, '{0}_mask.nii.gz'.format(op.basename(in_file).split('.')[0]))

    shutil.copyfile(gms_fn, gms_fn2)
    shutil.copyfile(mask_fn, mask_fn2)

    shutil.rmtree(work_dir)


def rs_firstlevel(unsmooth_fn, smooth_fn, roi_mask, output_dir, work_dir):

    import nipype.algorithms.modelgen as model  # model generation
    from niflow.nipype1.workflows.fmri.fsl import create_modelfit_workflow
    from nipype.interfaces import fsl as fsl
    from nipype.interfaces.base import Bunch

    meants = fsl.utils.ImageMeants()
    meants.inputs.in_file = unsmooth_fn
    meants.inputs.mask = roi_mask
    meants.inputs.out_file = op.join(work_dir, '{0}_{1}.txt'.format(unsmooth_fn.split('.')[0], op.basename(roi_mask).split('.')[0]))
    meants.cmdline
    meants.run()

    roi_ts = np.atleast_2d(np.loadtxt(op.join(work_dir, '{0}_{1}.txt'.format(unsmooth_fn.split('.')[0], op.basename(roi_mask).split('.')[0]))))
    subject_info = Bunch(conditions=['mean'], onsets=[list(np.arange(0,0.72*len(roi_ts[0]),0.72))], durations=[[0.72]], amplitudes=[np.ones(len(roi_ts[0]))], regressor_names=['roi'], regressors=[roi_ts[0]])

    level1_workflow = pe.Workflow(name='level1flow')

    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'subjectinfo']),
                        name='inputspec')

    modelspec = pe.Node(model.SpecifyModel(), name="modelspec")
    modelspec.inputs.input_units = 'secs'
    modelspec.inputs.time_repetition = 0.72
    modelspec.inputs.high_pass_filter_cutoff = 0

    modelfit = create_modelfit_workflow()
    modelfit.get_node('modelestimate').inputs.smooth_autocorr = False
    modelfit.get_node('modelestimate').inputs.autocorr_noestimate = True
    modelfit.get_node('modelestimate').inputs.mask_size = 0
    modelfit.inputs.inputspec.interscan_interval = 0.72
    modelfit.inputs.inputspec.bases = {'none': {'none': None}}
    modelfit.inputs.inputspec.model_serial_correlations = False
    modelfit.inputs.inputspec.film_threshold = 1000
    contrasts = [['corr', 'T', ['mean', 'roi'], [0,1]]]
    modelfit.inputs.inputspec.contrasts = contrasts

    """
    This node will write out image files in output directory
    """
    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    level1_workflow.connect(
        [(inputnode, modelspec, [('func', 'functional_runs')]),
         (inputnode, modelspec, [('subjectinfo', 'subject_info')]),
         (modelspec, modelfit, [('session_info', 'inputspec.session_info')]),
         (inputnode, modelfit, [('func', 'inputspec.functional_data')]),
         (modelfit, datasink, [('outputspec.copes','copes'),
                               ('outputspec.varcopes','varcopes'),
                               ('outputspec.dof_file','dof_file'),
                               ('outputspec.zfiles','zfiles')])])

    level1_workflow.inputs.inputspec.func = smooth_fn
    level1_workflow.inputs.inputspec.subjectinfo = subject_info
    level1_workflow.base_dir = work_dir

    level1_workflow.run()

    #copy data to directory
    shutil.rmtree(op.join(work_dir, 'level1flow'))
    files_to_copy = glob(op.join(work_dir, '*', '_modelestimate0', '*'))
    for tmp_fn in files_to_copy:
        shutil.copy(tmp_fn, output_dir)

    shutil.rmtree(work_dir)


def rs_secondlevel(copes, varcopes, dofs, output_dir, work_dir):

    from nipype.workflows.fmri.fsl.estimate import create_fixed_effects_flow
    from nipype.interfaces.fsl import Info

    level2workflow = pe.Workflow(name="level2workflow")
    level2workflow.base_dir = work_dir

    fixedfx = create_fixed_effects_flow()
    fixedfx.inputs.inputspec.copes = copes
    fixedfx.inputs.inputspec.varcopes = varcopes
    fixedfx.inputs.inputspec.dof_files = dofs
    fixedfx.inputs.l2model.num_copes = len(dofs)
    fixedfx.inputs.flameo.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    level2workflow.connect([(fixedfx, datasink, [('outputspec.copes','copes'),
                                                 ('outputspec.varcopes','varcopes'),
                                                 ('outputspec.zstats','zstats')])])



    level2workflow.run()

    #copy data to directory
    shutil.rmtree(op.join(work_dir, 'level2workflow'))
    files_to_copy = glob(op.join(work_dir, '*', '_flameo0', '*'))
    for tmp_fn in files_to_copy:
        shutil.copy(tmp_fn, output_dir)

    shutil.rmtree(work_dir)


def rs_grouplevel(copes, varcopes, output_dir, work_dir):

    from nipype.interfaces.fsl.model import MultipleRegressDesign
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

    model = pe.Node(interface=MultipleRegressDesign(), name='model')
    model.inputs.contrasts = [['mean', 'T', ['roi'], [1]]]
    model.inputs.regressors = dict(roi=np.ones(len(copes)).tolist())

    flameo = pe.Node(interface=FLAMEO(), name='flameo')
    flameo.inputs.cope_file = op.join(work_dir, 'cope.nii.gz')
    flameo.inputs.var_cope_file = op.join(work_dir, 'varcope.nii.gz')
    flameo.inputs.run_mode = 'flame1'
    flameo.inputs.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

    grplevelworkflow.connect(model, 'design_con', flameo, 't_con_file')
    grplevelworkflow.connect(model, 'design_grp', flameo, 'cov_split_file')
    grplevelworkflow.connect(model, 'design_mat', flameo, 'design_file')

    smoothest = pe.Node(SmoothEstimate(), name='smooth_estimate')
    grplevelworkflow.connect(flameo, 'zstats', smoothest, 'zstat_file')
    smoothest.inputs.mask_file = Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

    cluster = pe.Node(Cluster(), name='cluster')
    ptoz = pe.Node(PtoZ(), name='ptoz')
    grplevelworkflow.connect(smoothest, 'resels', cluster, 'resels')
    grplevelworkflow.connect(smoothest, ('resels', calcres), ptoz, 'resels')
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
            smooth_fn = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt, '{0}_smooth.nii.gz'.format(op.basename(nii_fn).split('.')[0]))

            if not op.isfile(smooth_fn):

                tmp_output_dir = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt)
                if not op.isdir(tmp_output_dir):
                    os.makedirs(tmp_output_dir)
                nii_work_dir = op.join(work_dir, 'rsfc', ppt, op.basename(nii_fn).split('.')[0])
                rs_preprocess(nii_fn, 4, nii_work_dir, tmp_output_dir)

            #run analysis
            tmp_output_dir = op.join(rs_data_dir, 'derivatives', coords_str, ppt, op.basename(nii_fn).split('.')[0])
            if not op.isdir(tmp_output_dir):
                os.makedirs(tmp_output_dir)
                nii_work_dir = op.join(work_dir, 'rsfc', coords_str, ppt, op.basename(nii_fn).split('.')[0])
                rs_firstlevel(nii_fn, smooth_fn, roi_mask_fn, tmp_output_dir, nii_work_dir)

        output_dir = op.join(rs_data_dir, 'derivatives', coords_str, ppt)
        if not op.isfile(op.join(output_dir, 'zstat1.nii.gz')):

            if len(nii_files)>1:

                copes = [sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, ppt, '*', 'cope*.nii.gz')))]
                varcopes = [sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, ppt, '*', 'varcope*.nii.gz')))]
                dofs = sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, ppt, '*', 'dof')))
                output_dir = op.join(rs_data_dir, 'derivatives', coords_str, ppt)
                nii_work_dir = op.join(work_dir, 'rsfc', coords_str, ppt)
                rs_secondlevel(copes, varcopes, dofs, output_dir, nii_work_dir)

            else:

                stat_files = sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, ppt, '*', '*.nii.gz')))
                for tmp_stat_file in stat_files:
                    shutil.copy(tmp_fn, output_dir)

    copes = sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, '*', 'cope*.nii.gz')))
    varcopes = sorted(glob(op.join(rs_data_dir, 'derivatives', coords_str, '*', 'varcope*.nii.gz')))

    output_dir = op.join(rs_data_dir, 'derivatives', coords_str)
    nii_work_dir = op.join(work_dir, 'rsfc', coords_str)
    if not op.isdir(nii_work_dir):
        os.makedirs(nii_work_dir)
    rs_grouplevel(copes, varcopes, output_dir, nii_work_dir)
