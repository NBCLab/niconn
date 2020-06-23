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
    gms_fn = glob(op.join(work_dir, 'gms', '_meanscale0', '*_gms.nii.gz'))
    mask_fn = glob(op.join(work_dir, 'mask', '_dilatemask0', '*_dil.nii.gz'))
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

    from nipype.workflows.fmri.fsl.estimate import create_fixed_effects_workflow

    level2workflow = pe.Workflow(name="level2workflow")
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['copes',
                                                                 'varcopes',
                                                                 'dofs']),
                        name='inputspec')

    fixedfx = create_fixed_effects_workflow()

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    level2workflow.connect([(inputnode, fixedfx, [('copes', 'inputs.inputspec.copes')]),
                            (inputnode, fixedfx, [('varcopes', 'inputs.inputspec.varcopes')]),
                            (inputnode, fixedfx, [('dofs', 'inputs.inputspec.dof_files')]),
                            (fixedfx, datasink, [('outputspec.copes','copes'),
                                                 ('outputspec.varcopes','varcopes'),
                                                 ('outputspec.zfiles','zfiles')])])

    level2workflow.inputs.inputspec.copes = [copes]
    level2workflow.inputs.inputspec.varcopes = [varcopes]
    level2workflow.inputs.inputspec.dofs = [dofs]
    level2workflow.base_dir = work_dir

    level2_workflow.run()

    #copy data to directory
    shutil.rmtree(op.join(work_dir, 'level2flow'))
    files_to_copy = glob(op.join(work_dir, '*', '_flameo0', '*'))
    for tmp_fn in files_to_copy:
        shutil.copy(tmp_fn, output_dir)

    shutil.rmtree(work_dir)


def rs_grouplevel(copes, varcopes, output_dir, work_dir):

    from nipype.interfaces.fsl.model import MultipleRegressDesign
    from nipype.interfaces.fsl.model import FLAMEO
    from nipype.interfaces.fsl.model import SmoothEstimate
    from interfaces import Cluster
    from nipype.interfaces.fsl.utils import Merge
    from nipype.interfaces.fsl.Info import standard_image
    from interfaces import PtoZ

    def calcres(smoothest_input):
        resels = int(smoothest_input[0]/smoothest_input[1])

    grplevelworkflow = pe.Workflow(name="grplevelworkflow")

    merger = Merge()
    merger.inputs.dimensions = 't'
    merger.inputs.in_files = copes
    merger.inputs.merged_file = op.join(work_dir, 'cope.nii.gz')
    merger.run()

    merger.inputs.in_files = varcopes
    merger.inputs.merged_file = op.join(work_dir, 'varcope.nii.gz')
    merger.run()

    model = MultipleRegressDesign()
    model.inputs.contrasts = [['mean', 'T', ['roi'], [1]]]
    model.intputs.regressors = dict(roi=np.ones(len(copes)))

    flameo = pe.Node(interface=FLAMEO(), name='flameo')
    flameo.inputs.cope_file = op.join(work_dir, 'cope.nii.gz')
    flameo.inputs.var_cope_file = op.join(work_dir, 'varcope.nii.gz')
    flameo.inputs.run_mode = 'flame1'
    flameo.inputs.mask = standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

    grplevelworkflow.connect(model, 'design_con', flameo, 'inputs.t_con_file')
    grplevelworkflow.connect(model, 'design_grp', flameo, 'inputs.cov_split_file')
    grplevelworkflow.connect(model, 'design_mat', flameo, 'inputs.design_file')

    smoothest = Node(SmoothEstimate(), name='smooth_estimate')
    grplevelworkflow.connect(flameo, 'zstats', smoothest, 'zstat_file')
    smoothest.inputs.mask_file = mask_file

    cluster = Node(Cluster(), name='cluster')
    grplevelworkflow.connect(smoothest, 'resels', cluster, 'resels')
    grplevelworkflow.connect(smoothest, (['volume', 'resels'], calcres), ptoz, 'resels')
    grplevelworkflow.connect(ptoz, 'zstat', cluster, 'threshold')
    cluster.inputs.connectivity = 26
    cluster.inputs.out_threshold_file = True
    cluster.inputs.out_index_file = True
    cluster.inputs.out_localmax_txt_file = True
    cluster.inputs.voxthresh = True

    grplevelworkflow.connect(flame, 'zstats', cluster, 'in_file')

    datasink = pe.Node(nio.DataSink(), name='sinker')
    datasink.inputs.base_directory = work_dir

    grplevelworkflow.connect(flameo, 'outputs')

def rs_workflow(rs_data_dir, pid_file, roi_prefix, roi_mask, work_dir):

    from nipype.interfaces.base import Bunch

    os.makedirs(op.join(work_dir, 'rsfc'))

    #get participants
    ppt_df = pandas.read_csv(pid_file, sep='/t')
    for ppt in ppt_df['participant_id']:
        nii_files = os.listdir(op.join(rs_data_dir, ppt, 'func'))
        for nii_fn in nii_files:

            #check to see if smoothed data exists
            unsmooth_fn = op.join(rs_data_dir, ppt, 'func', nii_fn)
            smooth_fn = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt, '{0}_smooth.nii.gz'.format(nii_fn.split('.')[0]))

            if not op.isfile(smooth_fn):

                output_dir = op.join(rs_data_dir, 'derivatives', 'smoothed', ppt)
                if not op.isdir(output_dir):
                    os.makedirs(output_dir)
                nii_work_dir = op.join(work_dir, 'rsfc', ppt, nii_fn.split('.')[0])
                rs_preprocess(unsmooth_fn, 4, nii_work_dir, output_dir)

            #run analysis
            output_dir = op.join(rs_data_dir, 'derivatives', roi_prefix, ppt, nii_fn.split('.')[0])
            nii_work_dir = op.join(work_dir, 'rsfc', roi_prefix, ppt, nii_fn.split('.')[0])
            rs_firstlevel(unsmooth_fn, smooth_fn, roi_mask, output_dir, nii_work_dir)

        if len(nii_files)>1:

            copes = sorted(glob(op.join(rs_data_dir, 'derivatives', roi_prefix, ppt, '*', 'cope*.nii.gz')))
            varcopes = sorted(glob(op.join(rs_data_dir, 'derivatives', roi_prefix, ppt, '*', 'varcope*.nii.gz')))
            dofs = sorted(glob(op.join(rs_data_dir, 'derivatives', roi_prefix, ppt, '*', 'dof')))
            output_dir = op.join(rs_data_dir, 'derivatives', roi_prefix, ppt)
            nii_work_dir = op.join(work_dir, 'rsfc', roi_prefix, ppt)
            rs_secondlevel(copes, varcopes, dofs, output_dir, nii_work_dir)

    copes = sorted(glob(op.join(rs_data_dir, 'derivatives', roi_prefix, ppt, 'cope*.nii.gz')))
    varcopes = sorted(glob(op.join(rs_data_dir, 'derivatives', roi_prefix, ppt, 'varcope*.nii.gz')))

    output_dir = op.join(rs_data_dir, 'derivatives', roi_prefix)
    nii_work_dir = op.join(work_dir, 'rsfc', roi_prefix)
    rs_grouplevel(copes, varcopes, output_dir, nii_work_dir)
