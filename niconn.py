import os
import os.path as op
import pandas as pd
import numpy as np
import nibabel as nib
import requests
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

    os.makedirs(output_dir, exist_ok=True)

    macm_out_dir = op.join(output_dir, 'macm')
    os.makedirs(macm_out_dir, exist_ok=True)
    rsfc_out_dir = op.join(output_dir, 'rsfc')
    os.makedirs(rsfc_out_dir, exist_ok=True)

    if ns_data_dir is None:
        ns_data_dir = op.join(output_dir, 'neurosynth_dataset')

    os.makedirs(ns_data_dir, exist_ok=True)

    if rs_data_dir is None:
        rs_data_dir = op.join(output_dir, 'hcp1200_resting-state')

    if not op.isdir(rs_data_dir):
        os.makedirs(rs_data_dir)
        download_hcp=True

    if work_dir is None:
        work_dir = op.join(output_dir, 'niconn-work')

    os.makedirs(work_dir, exist_ok=True)

    macm_work_dir = op.join(work_dir, 'macm')
    os.makedirs(macm_work_dir, exist_ok=True)
    rsfc_work_dir = op.join(work_dir, 'rsfc')
    os.makedirs(rsfc_work_dir, exist_ok=True)

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
        macm_get_request = requests.get('https://niconn.s3.amazonaws.com/macm/{coords_str}/{coords_str}_{macm_suffix}.nii.gz'.format(coords_str=coords_str, macm_suffix=macm_suffix))

        if macm_get_request.status_code == 404:

            tmp_work_dir = op.join(macm_work_dir, coords_str)
            os.makedirs(tmp_work_dir, exist_ok=True)

            macm_workflow(x=vox[0], y=vox[1], z=vox[2], ns_data_dir=ns_data_dir, output_dir=tmp_work_dir)

            suffix = ['z', 'logp_level-voxel_corr-FWE_method-permutation', 'logp_level-cluster_corr-FWE_method-permutation']
            for tmp_suffix in suffix:
                tmp_fn = op.join(tmp_work_dir, '{coords_str}_{suffix}.nii.gz'.format(coords_str=coords_str, suffix=tmp_suffix))
                aws_fn = op.join('macm', coords_str, op.basename(tmp_fn))
                macm_put_request = requests.post('https://niconn.s3.amazonaws.com/', files={'file': open(tmp_fn, 'rb')}, data={'key': aws_fn})

            os.rmtree(tmp_work_dir)

        macm_get_request = requests.get('https://niconn.s3.amazonaws.com/macm/{coords_str}/{coords_str}_{macm_suffix}.nii.gz'.format(coords_str=coords_str, macm_suffix=macm_suffix))

        with open(op.join(macm_work_dir, '{coords_str}_{suffix}.nii.gz'.format(coords_str=coords_str, suffix=macm_suffix)), 'wb') as f:
            f.write(macm_get_request.content)

        #now resting-state
        rsfc_get_request = requests.get('https://niconn.s3.amazonaws.com/rsfc/{coords_str}/{coords_str}_{rsfc_suffix}.nii.gz'.format(coords_str=coords_str, rsfc_suffix=rsfc_suffix))

        if rsfc_get_request.status_code == 404:

            tmp_work_dir = op.join(rsfc_work_dir, coords_str)
            os.makedirs(tmp_work_dir, exist_ok=True)

            rs_workflow(x=vox[0], y=vox[1], z=vox[2], rs_data_dir=rs_data_dir, output_dir=tmp_work_dir)

            suffix = ['tstat1', 'tstat1_thr001', 'vox_corrp']
            for tmp_suffix in suffix:
                tmp_fn = op.join(tmp_work_dir, '{coords_str}_{suffix}.nii.gz'.format(coords_str=coords_str, suffix=tmp_suffix))
                aws_fn = op.join('rsfc', coords_str, op.basename(tmp_fn))
                macm_put_request = requests.post('https://niconn.s3.amazonaws.com/', files={'file': open(tmp_fn, 'rb')}, data={'key': aws_fn})

            os.rmtree(tmp_work_dir)

        rsfc_get_request = requests.get('https://niconn.s3.amazonaws.com/rsfc/{coords_str}/{coords_str}_{rsfc_suffix}.nii.gz'.format(coords_str=coords_str, rsfc_suffix=rsfc_suffix))

        with open(op.join(rsfc_work_dir, '{coords_str}_{suffix}.nii.gz'.format(coords_str=coords_str, suffix=rsfc_suffix)), 'wb') as f:
            f.write(rsfc_request.content)

    #evaluate MACMs now
    macm_img_list=sorted(glob(op.join(macm_work_dir, '*_{suffix}.nii.gz'.format(suffix=macm_suffix))))

    fishers_workflow(img_list=macm_img_list, prefix='ibma_{prefix}'.format(prefix=prefix), output_dir=macm_out_dir)

    macm(prefix='true_{prefix}'.format(prefix=prefix), mask=mask_img, output_dir=macm_out_dir, ns_data_dir=ns_data_dir)

    #evalute rsFC now
    rsfc_img_list=sorted(glob(op.join(rsfc_work_dir, '*_{suffix}.nii.gz'.format(suffix=rsfc_suffix))))
