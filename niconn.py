import os
import os.path as op
import pandas as pd
import numpy as np
import nibabel as nib
import boto3
from connectivity.macm import macm_workflow
from connectivity.macm import macm
from connectivity.rsfc import rs_workflow

from glob import glob
from ibma import fishers_workflow

def niconn_workflow(mask_img=None, prefix=None, output_dir=None, ns_data_dir=None, rs_data_dir=None, work_dir=None):

    if mask_img is None or not op.isfile(mask_img):
        raise Exception('A valid NIFTI file is required!')

    if prefix is None:
        prefix = op.basename(mask_img).split('.')[0]

    if output_dir is None:
        output_dir = op.join(op.dirname(op.abspath(mask_img)), prefix)

    if not op.isdir(output_dir):
        os.makedirs(output_dir)

    macm_out_dir = op.join(output_dir, 'macm')
    os.makedirs(macm_out_dir)
    rsfc_out_dir = op.join(output_dir, 'rsfc')
    os.makedirs(rsfc_out_dir)

    if ns_data_dir is None:
        ns_data_dir = op.join(output_dir, 'neurosynth_dataset')

    if not op.isdir(ns_data_dir):
        os.makedirs(ns_data_dir)

    if rs_data_dir is None:
        rs_data_dir = op.join(output_dir, 'hcp1200_resting-state')

    if not op.isdir(rs_data_dir):
        os.makedirs(rs_data_dir)
        download_hcp=True

    if work_dir is None:
        work_dir = op.join(output_dir, 'niconn-work')

    if not op.isdir(work_dir):
        os.makedirs(work_dir)

    macm_work_dir = op.join(work_dir, 'macm')
    os.makedirs(macm_work_dir)
    rsfc_work_dir = op.join(work_dir, 'rsfc')
    os.makedirs(rsfc_work_dir)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket('niconn')
    all_items = [obj.key for obj in bucket.objects.all()]
    all_items.remove('rsfc/')
    all_items.remove('macm/')
    rsfc_items = [x for x in all_items if 'rsfc' in x]
    macm_items = [x for x in all_items if 'macm' in x]

    img = nib.load(mask_img)

    inds = np.nonzero(img.get_data())
    inds = np.transpose(inds)

    macm_suffix = 'logp_level-voxel_corr-FWE_method-permutation'
    rsfc_suffix = 'thresh_zstat'
    
    for tmpind in inds:
        vox = np.dot(img.affine, np.append(tmpind, 1))
        vox = vox[0:3].astype(int)

        coords_str = '{x}_{y}_{z}'.format(x=str(vox[0]), y=str(vox[1]), z=str(vox[2]))

        #macms first
        if 'macm/{coords_str}/{coords_str}_z.nii.gz'.format(coords_str=coords_str) not in macm_items:

            tmp_work_dir = op.join(macm_work_dir, coords_str)
            os.makedirs(tmp_work_dir)

            macm_workflow(x=vox[0], y=vox[1], z=vox[2], ns_data_dir=ns_data_dir, output_dir=tmp_work_dir)

            suffix = ['z', 'logp_level-voxel_corr-FWE_method-permutation', 'logp_level-cluster_corr-FWE_method-permutation']
            for tmp_suffix in suffix:
                tmp_fn = op.join(tmp_work_dir, '{coords_str}_{suffix}.nii.gz'.format(coords_str=coords_str, suffix=tmp_suffix))
                aws_fn = op.join('macm', coords_str, op.basename(tmp_fn))
                s3.Bucket('niconn').upload_file(tmp_fn, aws_fn)

            os.rmtree(tmp_work_dir)

        s3.Bucket('niconn').download_file(op.join('macm', coords_str, '{coords_str}_{suffix}.nii.gz'.format(coords_str=coords_str, suffix=macm_suffix)), op.join(macm_work_dir, '{coords_str}_{suffix}.nii.gz'.format(coords_str=coords_str, suffix=macm_suffix)))

        if 'rsfc/{coords_str}/{coords_str}_z.nii.gz'.format(coords_str=coords_str) not in rsfc_items:

            tmp_work_dir = op.join(rsfc_work_dir, coords_str)
            os.makedirs(tmp_work_dir)

            rs_workflow(x=vox[0], y=vox[1], z=vox[2], rs_data_dir=rs_data_dir, output_dir=tmp_work_dir)

            os.rmtree(tmp_work_dir)

    #evaluate MACMs now
    macm_img_list=sorted(glob(op.join(macm_work_dir, '*_{suffix}.nii.gz'.format(suffix=macm_suffix))))

    fishers_workflow(img_list=macm_img_list, prefix='ibma_{prefix}'.format(prefix=prefix), output_dir=macm_out_dir)

    macm(prefix='true_{prefix}'.format(prefix=prefix), mask=mask_img, output_dir=macm_out_dir, ns_data_dir=ns_data_dir)

    #evalute rsFC now
    rsfc_img_list=sorted(glob(op.join(rsfc_work_dir, '*_{suffix}.nii.gz'.format(suffix=rsfc_suffix))))
