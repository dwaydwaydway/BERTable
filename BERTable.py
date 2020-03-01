import warnings

import numpy as np
import pickle as pkl
import torch
from tqdm import tqdm

from modules.model import Model
from modules.dataset import create_dataloader, transfer
from modules.vocab import Vocab
from modules.logger import create_logger

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")


class BERTable():
    def __init__(
            self,
            df, column_type,
            embedding_dim=5, n_layers=5, dim_feedforward=100, n_head=5,
            dropout=0.15, ns_exponent=0.75, share_category=False, use_pos=False, device='cpu'):

        self.logger = create_logger(name="BERTable")

        self.col_type = {'numerical': [], 'categorical': [], 'vector': []}
        for i, data_type in enumerate(column_type):
            self.col_type[data_type].append(i)

        self.embedding_dim = embedding_dim
        self.use_pos = use_pos
        self.device = device

        self.logger.info(f'[-] Building Vocab ...')
        self.vocab = Vocab(
            df,
            self.col_type,
            share_category,
            ns_exponent)

        vocab_size = {
            'numerical': len(self.vocab.item2idx['numerical']),
            'categorical': len(self.vocab.item2idx['categorical'])}

        vector_dims = [np.shape(df[col])[1]
                       for col in self.col_type['vector']]
        tab_len = len(column_type)
        self.model = Model(
            vocab_size, self.col_type, use_pos,
            vector_dims, embedding_dim, dim_feedforward, tab_len,
            n_layers, n_head, dropout).to(device)

    def fit(
            self,
            df,
            max_epochs=3, lr=1e-4,
            lr_weight={'numerical': 0.33, 'categorical': 0.33, 'vector': 0.33},
            loss_clip=[0, 100],
            n_sample=4, mask_rate=0.15, replace_rate=0.8,
            batch_size=32, shuffle=True, num_workers=1):

        self.logger.info("[-] Start Pretraining")
        self.model.loss_clip = loss_clip

        data = self.vocab.convert(df, num_workers)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(lr))

        process_bar = tqdm(
            range(max_epochs),
            desc=f"[Progress]",
            total=max_epochs,
            position=0)

        metric_bar = tqdm(
            [0],
            desc=f"[Metric]",
            bar_format="{desc}{postfix}",
            leave=False,
            position=2)

        for epoch in process_bar:

            generator = create_dataloader(
                data, self.col_type, self.vocab,
                self.embedding_dim, self.use_pos,
                batch_size, num_workers,
                mask_rate=mask_rate,
                replace_rate=replace_rate,
                n_sample=n_sample,
                shuffle=shuffle)

            epoch_bar = tqdm(
                generator,
                desc=f"[Epoch]",
                leave=False,
                position=1)

            loss_history = {'numerical': [], 'categorical': [], 'vector': []}

            for batch_data in epoch_bar:

                batch_data = transfer(batch_data, self.device)
                _, losses = self.model.forward(batch_data, mode='train')

                loss = sum([losses[data_type] / len(self.col_type[data_type]) * lr_weight[data_type]
                            for data_type in self.col_type if len(self.col_type[data_type]) > 0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                display = ''
                for types in losses:
                    loss_history[types].append(losses[types].item())
                    display += f'{types}: {np.mean(loss_history[types]):5.2f} '
                metric_bar.set_postfix_str(display)

            process_bar.write(f'[Log] Epoch {epoch:0>2d}| ' + display)
            epoch_bar.close()
        metric_bar.close()
        process_bar.close()

    def transform(self, df, batch_size=32, num_workers=1):
        self.logger.info("[-] Start Transforming")
        data = self.vocab.convert(df, num_workers)
        prcess_bar = tqdm(
            range(1),
            desc=f"[Progress]",
            position=0)

        generator = create_dataloader(
            data, self.col_type, self.vocab,
            self.embedding_dim, self.use_pos,
            batch_size, num_workers,
            mask_rate=0,
            n_sample=1,
            shuffle=False)

        process_bar = tqdm(
            generator,
            desc=f"[Process]",
            leave=False,
            position=1)

        df_t = []
        for batch_data in process_bar:
            batch_data = transfer(batch_data, self.device)
            feature = self.model.forward(batch_data, mode='test')
            df_t += list(feature)

        process_bar.close()
        return df_t

    def save(self, model_path='model.ckpt', vocab_path='vocab.pkl'):
        torch.save(self.model.state_dict(), model_path)
        with open(vocab_path, 'wb') as file:
            pkl.dump(self.vocab, file)