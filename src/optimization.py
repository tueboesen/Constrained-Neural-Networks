import time

import mlflow.pytorch
import torch
import torch.nn.utils.parametrize as P
from e3nn import o3
from tqdm import tqdm

from src.loss import loss_eq, loss_mim
from src.vizualization import plot_pendulum_snapshot, plot_pendulum_snapshot_custom

def log_parameters(c):
    """
    We log all the parameters in the configuration using mlflow.
    We assume that each parameter is part of a configuration group and does not have any configuration subgroups. #TODO this is something we should make more flexible at some point.
    Hence a parametere will be logged as {group_key}.{key}
    """
    for key,val in c.items():
        for keyi,vali in c[key].items():
            mlflow.log_param(f"{key}.{keyi}",vali)


def optimize_model(c,model,dataloaders,optimizer,loss_fnc):
    """
    Used to optimize and run inference of models.
    Tracking is done with MLFlow.

    c :: configuration OmegaConf
    model: pytorch model
    dataloaders: A dict containing various dataloaders saved with keys such as 'train' 'val' 'test'.
    """

    names = ['train', 'val']
    mlflow.set_tracking_uri(c.logging.mlflow_folder)
    with mlflow.start_run(run_name=c.run.name) as run:
        # artifact_path = "models"
        log_parameters(c)
        metrics = Metrics(name='train')
        for epoch in range(c.run.epochs):
            for name in names:
                if name in dataloaders:
                    loss = run_model(epoch,c,model,dataloaders[name],optimizer,loss_fnc,metrics,type=name)
        if 'test' in dataloaders:
            loss = run_model(0, c, model, dataloaders['test'], optimizer, loss_fnc, metrics, type='test')
        mlflow.pytorch.log_state_dict(model.state_dict(), 'models')
        mlflow.pytorch.get_default_pip_requirements()
        # mlflow.pytorch.save_model(model,'modelsdfs')
        # mlflow.pytorch.log_model(model.to(device='cpu'), "models")
        # mlflow.pytorch.get_default_conda_env()
        # mlflow.pytorch.get_default_pip_requirements()
    return loss

class Metrics:
    """
    The metrics which we want to save to mlflow.
    """
    def __init__(self,name,report_every_n_step=0):
        self.report_every_n_step = report_every_n_step
        self.reset(name)

    def reset(self,name):
        self.name = name
        self.cv_mean = 0.0
        self.cv_max = 0.0
        self.reg = 0.0
        self.loss_predict = 0.0
        self.loss = 0.0
        self.mae_r = 0.0
        self.mae_v = 0.0
        self.counter = 0
        self.epoch = 0

    def update(self,epoch,cv_mean,cv_max,reg,loss_predict,loss,mae_r,mae_v):
        self.epoch = epoch
        self.cv_mean += cv_mean.item()
        self.cv_max += cv_max.item()
        self.reg += reg.item()
        self.loss_predict += loss_predict.item()
        self.loss += loss.item()
        self.mae_r += mae_r.item()
        self.mae_v += mae_v.item()
        self.counter += 1
        self.report()

    def report(self,end_of_epoch=False):
        if end_of_epoch or ((self.report_every_n_step > 0) and ((self.counter % self.report_every_n_step) == 0)):
            # mlflow.log_metric("epoch",self.epoch)
            mlflow.log_metric(f"{self.name}_cv_mean",self.cv_mean/self.counter,step=self.epoch)
            mlflow.log_metric(f"{self.name}_cv_max",self.cv_max/self.counter,step=self.epoch)
            mlflow.log_metric(f"{self.name}_reg",self.reg/self.counter,step=self.epoch)
            mlflow.log_metric(f"{self.name}_loss_predict",self.loss_predict/self.counter,step=self.epoch)
            mlflow.log_metric(f"{self.name}_loss",self.loss/self.counter,step=self.epoch)
            mlflow.log_metric(f"{self.name}_mae_r",self.mae_r/self.counter,step=self.epoch)
            mlflow.log_metric(f"{self.name}_mae_v",self.mae_v/self.counter,step=self.epoch)

def compute_mae(c,x_pred,x_target,rscale,vscale):
    """
    Computes the Mean absolute error for both positions and velocities.
    rscale,vscale are used to scale the data back to original units
    """
    for key,idx in c.data.data_id.items():
        if key == 'r':
            pred = x_pred[...,idx]
            target = x_target[...,idx]
            mae_r = torch.mean(torch.norm((pred - target)*rscale,dim=1))
        elif key == 'v':
            pred = x_pred[...,idx]
            target = x_target[...,idx]
            mae_v = torch.mean(torch.norm((pred - target)*vscale,dim=1))
        else:
            raise(f"{key} data type not implemented for MAE calculation.")
    return mae_r, mae_v



def run_model(epoch,c,model,dataloader,optimizer,loss_fnc,metrics,type):
    """
    Runs the models for a single epoch.
    Supports both training and evaluation mode.
    """
    train = type == 'train'
    model.train(train)
    torch.set_grad_enabled(train)
    metrics.reset(name=type)
    for i, (Rin, Rout, Vin, Vout, z, m) in enumerate((pbar := tqdm(dataloader, desc=f"{type} Epoch: {epoch}"))):
        optimizer.zero_grad()

        t1 = time.time()
        batch, x, z_vec, m_vec, weights, x_target = dataloader.collate_vars(Rin, Rout, Vin, Vout, z, m)
        t2 = time.time()
        edge_src, edge_dst, wstatic = dataloader.generate_edges(batch,x,model.max_radius)
        t3 = time.time()
        with P.cached():
            x_pred, cv_mean,cv_max, reg,= model(x, batch, z_vec, edge_src, edge_dst,wstatic=wstatic,weight=weights)
        t4 = time.time()
        loss_pred = loss_fnc(x_pred,x_target,x,edge_src,edge_dst)
        loss = loss_pred + reg
        mae_r, mae_v = compute_mae(c,x_pred,x_target,dataloader.dataset.rscale,dataloader.dataset.vscale)
        metrics.update(epoch,cv_mean,cv_max,reg,loss_pred,loss,mae_r,mae_v)
        t5 = time.time()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
        pbar.set_postfix(loss=metrics.loss/metrics.counter)
        t6 = time.time()
        # print(f"{t2-t1:2.2f},{t3-t2:2.2f},{t4-t3:2.2f},{t5-t4:2.2f},{t6-t5:2.2f}")
    metrics.report(end_of_epoch=True)
    return metrics.loss


