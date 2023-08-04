import os
import wandb
import hydra
from tqdm import tqdm
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

from dataset import MetasurfaceDataset
from losses import BidirectionalLoss
from networks import build_inn_model
from trainer import train_epoch, test_epoch
from plotter import Plotter


def wandb_init(cfg: DictConfig):
    wandb.init(
        project='inn',
        group=cfg.exp_group,
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))


@hydra.main(version_base=None, config_path='configs', config_name='meta-atom')
def main(cfg: DictConfig) -> None:
    """ I. Logger. """
    if cfg['enable_wandb']:
        wandb_init(cfg)

    """ II. Datasets. """
    train_dataset = MetasurfaceDataset(dataset_dir=cfg['data_dir'], split='train')
    test_dataset = MetasurfaceDataset(dataset_dir=cfg['data_dir'], split='test')
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train_batch_size'],
                              num_workers=2,
                              drop_last=True,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['test_batch_size'],
                             num_workers=2,
                             drop_last=False,
                             shuffle=False)

    """ III. Model, criterion, optimizer, scheduler and plotter. """
    # Create the model in selected device.
    model = build_inn_model(**cfg['model'])
    model = model.to(cfg['device'])

    # Create criterion
    criterion = BidirectionalLoss(**cfg['criterion'])

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), **cfg['train']['optimizer'])

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **cfg['train']['scheduler'])

    # Create plotter
    plotter = Plotter()

    """ V. Training. """
    # Change to working directory
    os.chdir(HydraConfig.get().runtime.output_dir)

    for epoch in tqdm(range(cfg['train']['n_epoch'])):

        train_epoch_loss = train_epoch(cfg['data'], epoch, model, criterion, train_loader, optimizer, mode='train')
        test_epoch_loss = test_epoch(cfg['data'], epoch, model, criterion, test_loader, plotter, mode='train')

        scheduler.step()

        if cfg['enable_wandb']:
            wandb.log({
                'train': train_epoch_loss,
                'test': test_epoch_loss,
                'cov_z': plotter.get_cov_fig(),
                'hist_z': plotter.get_hist_fig(),
                'posterior': plotter.get_posterior()
            })

        if epoch % 10 == 0:
            torch.save({'opt': optimizer.state_dict(), 'net': model.state_dict()}, cfg['model_name'] + f'epoch{epoch}' + '.pt')

    torch.save({'opt': optimizer.state_dict(), 'net': model.state_dict()}, cfg['model_name'] + f'epoch{epoch}' + '.pt')

    """
    for epoch in tqdm(range(cfg['train']['init_epoch'], cfg['train']['init_epoch'] + cfg['train']['n_epoch'])):

        train_epoch_loss = train_epoch(cfg['data'], epoch, model, criterion, train_loader, optimizer, mode='train')
        test_epoch_loss = test_epoch(cfg['data'], epoch, model, criterion, test_loader, plotter, mode='train')

        scheduler.step()

        if cfg['enable_wandb']:
            wandb.log({
                'train': train_epoch_loss,
                'test': test_epoch_loss,
                'cov_z': plotter.get_cov_fig(),
                'hist_z': plotter.get_hist_fig(),
                'posterior': plotter.get_posterior()
            })
    
    torch.save({'opt': optimizer.state_dict(), 'net': model.state_dict()}, cfg['model_name'] + '.pt')
    """


if __name__ == '__main__':
    main()
