# sudo apt update
# sudo apt-get install postgresql-client nginx
# script for bootstrapping EC2 web server which skips all prompts
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
eval "$(/home/ubuntu/miniconda/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false
conda create -n insight python -y
conda activate insight
# pytorch
conda install -y pytorch torchvision cpuonly -c pytorch
# data science stuff (some of this will be redundant which is OK)
conda install -y numpy scipy pandas scikit-learn matplotlib seaborn
# web server
conda install -y flask gunicorn
pip install psycopg2-binary
# Might need to close out of session if conda env isn't showing as active
# https://certbot.eff.org/lets-encrypt/ubuntubionic-nginx