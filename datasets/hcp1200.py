import os
import os.path as op

def hcp1200_download(hcp_data_dir=None):

    from datalad.api import install

    if hcp_data_dir is None:
        raise Exception('A valid directory is required for downloading HCP data!')

    if not op.isdir(hcp_data_dir):
        os.mkdir(hcp_data_dir)
    os.chdir(hcp_data_dir)

    install(source='///hcp-openaccess')


def generate_random_pid(hcp_data_dir=None, n_pid=150):

    import random

    all_pid = os.listdir(op.join(hcp_data_dir))
    rand_pid = sorted(random.sample(all_pid, n_pid))
    return rand_pid


def download_ppt(hcp_data_dir=None, pid=None):

    from glob import glob
    from datalad.api import get
    from datalad.api import install

    os.chdir(hcp_data_dir)
    for tmp_pid in pid:
        install(op.join(hcp_data_dir, tmp_pid, 'MNINonLinear', 'Results'))
        tmp_rs_dir = sorted(glob(op.join(hcp_data_dir, tmp_pid, 'MNINonLinear', 'Results', 'rfMRI_*')))
        for tmp_rs_run in tmp_rs_dir:
            tmp_rs_run = tmp_rs_run.split('/')[-1]
            get(op.join(hcp_data_dir, tmp_pid, 'MNINonLinear', 'Results', tmp_rs_run, '{0}_hp2000_clean.nii.gz'.format(tmp_rs_run)))
