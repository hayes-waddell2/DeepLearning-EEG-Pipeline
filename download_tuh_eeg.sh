#!/bin/bash -1
#SBATCH --job-name=tuh-eeg-download
#SBATCH --account=eeg-cnn-lstm
#SBATCH --partition=debug
#SBATCH --output=/shared/rc/eeg-cnn-lstm/download_%j.log
#SBATCH --time=0-1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

. /tools/spack/share/spack/setup-env.sh
spack load rsynch@3.2.7

rsync -auvxL -e "ssh -i ~/.ssh/id_ed25519_tuh" \
    nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal/v3.0.1 \
    /shared/rc/eeg-cnn-lstm/data/
