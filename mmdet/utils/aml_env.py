import os


def get_aml_master_ip(master_name=None):
    if 'AZ_BATCHAI_JOB_MASTER_NODE_IP' in os.environ:
        return os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
    elif 'OMPI_MCA_orte_local_daemon_uri' in os.environ:
        return os.environ['OMPI_MCA_orte_local_daemon_uri'].split('/')[-1].split(':')[0].split(',')[-1]
        # DLTS_SD_ps0_IP
    else:
        print("ENVIRON:", os.environ)
        return None
