import os
import os.path as op
from datalad.api import install

def hcp1200_download(hcp_data_dir=None):

    if hcp_data_dir is None:
        raise Exception('A valid directory is required for downloading HCP data!')

    if not op.isdir(hcp_data_dir):
        os.mkdir(hcp_data_dir)
    os.chdir(hcp_data_dir)

    install(source='///hcp-openaccess')
