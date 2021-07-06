BERTable: Universal Representation Learning for Tabular data
===

## Requirements
* Python >= 3.7
* Numpy >= 1.17.4
* PyTorch >= 1.13.0
* tqdm >= 4.40.2

## Usage
```python
from BERTable import BERTable

# Read dataset
df = pd.read_csv('dataset.csv', header=None)
column_type = ['numerical', 'categorical', 'numerical', 'numerical', 'categorical'....]
df = df.values.tolist()

# Initialization
bertable = BERTable(
    df, column_type,
    embedding_dim=5, n_layers=5, dim_feedforward=100, n_head=5,
    dropout=0.15, ns_exponent=0.75, share_category=False, use_pos=False)

# Start self-supervised Pretraining
bertable.fit(
    df, 
    max_epochs=3, lr=1e-4,
    lr_weight={'numerical': 0.33, 'categorical': 0.33, 'vector': 0.33},
    loss_clip = [0, 100],
    n_sample=5, mask_rate=0.15, replace_rate=0.8, 
    batch_size=256, shuffle=True, num_workers=10)

# Feature Extraction
df_t = bertable.transform(df, batch_size=256, num_workers=10)
```
## Parameters
#### BERTable.BERTable
* <strong>df</strong> (list, required)
    > The data used for training. 
* <strong>column_type</strong> (list, required)
    > Specify the column types. 'numerical, 'categorical' or 'vector'.
* <strong>embedding_dim</strong> (int, default: 5)
    > Embedding dimension.
* <strong>n_layers</strong> (int, default: 5)
    > Number of transformer encoder layers.
* <strong>dim_feedforward</strong> (int, default: 100)
    > Hidden dimension of transformer encoder layers.
* <strong>n_head</strong> (int, default: 5)
    > The number of heads in the multiheadattention models.
* <strong>dropout</strong> (float, default: 0.15)
    > The dropout value.
* <strong>ns_exponent</strong> (float, default: 0.75)
    > The exponent used to shape the negative sampling distribution.
* <strong>share_category</strong> (bool, default: Fasle)
    > If True, same categorical data in different columns that share the same name will be treated as the same object.
* <strong>use_pos</strong> (bool, default: Fasle)
    > Whether or not to add positional embedding.

#### BERTable.BERTable.fit
* <strong>df</strong> (list, required)
    > The data used for training. 
* <strong>max_epochs</strong> (int, default: 3)
    > Number of epoch to train. 
* <strong>lr</strong> (float, default: 1e-4)
    > Learning rate for the optimizer. 
* <strong>lr_weight</strong> (dict, default: {'numerical': 0.33, 'categorical': 0.33, 'vector': 0.33})
    > Learning rate weight for each data type. 
* <strong>loss_clip</strong> (list, default: [0, 100])
    > Loss clipping for numerical data. 
* <strong>n_sample</strong> (int, default: 4)
    > Number negative samples to use.
* <strong>mask_rate</strong> (float, default: 0.15)
    > The masking probability.
* <strong>replace_rate</strong> (float, default: 0.8)
    > The masking probability.
* <strong>batch_size</strong> (int, default: 32)
    > The batch size.
* <strong>shuffle</strong> (bool, default: True)
    > Whether or not to shuffle data.
* <strong>num_workers</strong> (int, default: 1)
    > NUmber of workers.

## Experiments
Check ```exp``` folder for detail implimentatin of the experiments.


