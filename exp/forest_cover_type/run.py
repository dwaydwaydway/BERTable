import sys
sys.path.append('../..')
import pandas as pd
import ipdb
from box import Box
import warnings
warnings.filterwarnings("ignore")

from BERTable import BERTable
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(config):

    df = pd.read_csv(
        '/home/user/BERTable/exp/adult/data/adult.csv', header=None)
    column_type = ['numerical', 'categorical', 'numerical', 'categorical', 'numerical', 'categorical', 'categorical',
                   'categorical', 'categorical', 'categorical', 'numerical', 'numerical', 'numerical', 'categorical']
    df = df.values.tolist()
    
    bertable = BERTable(
        df, column_type,
        embedding_dim=10, n_layers=5, dim_feedforward=100, n_head=10,
        dropout=0.15, ns_exponent=0.75, share_category=False, use_pos=False, device=device)

    bertable.fit(
        df, 
        max_epochs=100, lr=1e-3,
        lr_weight={'numerical': 0.3, 'categorical': 0.3, 'vector': 0.33},
        loss_clip = [0, 100],
        n_sample=5, mask_rate=0.15, replace_rate=0.8, 
        batch_size=256, shuffle=True, num_workers=10)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        with open('config_template.yaml', 'r') as file:
            config = Box.from_yaml(file)
        main(config)
