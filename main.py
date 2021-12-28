import os
import copy
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from data import MultiViewDataset
from models import EMDL


class Experiment:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load dataset
        data_train = MultiViewDataset(train=True)
        data_valid = MultiViewDataset(train=False)
        num_classes = len(set(data_train.y))
        self.train_loader = torch.utils.data.DataLoader(data_train, batch_size=256, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=1024, shuffle=False)

        # Define model
        self.model = EMDL(
            len(data_train), len(data_valid),
            sample_shapes=[s.shape for s in data_train[0]['x'].values()],
            num_classes=num_classes,
            degrade=0.1
        ).to(self.device)

        # Define optimizer
        self.optimizer = torch.optim.SGD([
            {'params': self.model.inferences.parameters(), 'lr': 0.01, 'weight_decay': 1e-4},
            {'params': self.model.degrade_loss.parameters(), 'lr': 0.0001},
            {'params': self.model.evidences[0].parameters(), 'lr': 1000},
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=400, gamma=0.1)
        self.epochs = 500

    def train(self, saving_path=None):
        model = self.model
        best_valid_acc = 0.
        best_model_wts = model.state_dict()
        write_time = time.strftime("%m%d_%H%M%S", time.localtime())
        writer = SummaryWriter(log_dir=f'./tb/{write_time.replace("_", "/")}')
        batch = next(iter(self.train_loader))
        test_index, test_x, test_y = batch['index'], batch['x'], batch['y']
        for epoch in range(self.epochs):
            model.train()
            train_loss, correct, num_samples = 0, 0, 0
            for batch in self.train_loader:
                index = batch['index']
                x = batch['x']
                y = batch['y']
                """
                Training:
                Update evidence.
                Update U of degradation loss function.
                Update inference net.
                """
                for opt_ord in [2, 1, 0]:
                    for i, pg in enumerate(self.optimizer.param_groups):
                        for p in pg['params']:
                            p.requires_grad = (i == opt_ord)
                    ret = model(index, x, y, epoch)
                    self.optimizer.zero_grad()
                    ret['loss'].mean().backward()
                    self.optimizer.step()
                # Validation
                ret = model(index, x, y, epoch)
                view_e, fusion_e, loss, loss_c, loss_d = (ret[i] for i in ['view_e', 'fusion_e', 'loss', 'loss_c', 'loss_d'])
                train_loss += loss.mean().item() * len(y)
                correct += torch.sum(model.ds_combine(view_e).argmax(dim=-1).eq(y.to(self.device))).item()
                num_samples += len(y)
            self.scheduler.step()
            train_loss = train_loss / num_samples
            train_acc = correct / num_samples
            valid_acc = self.validate()
            if best_valid_acc < valid_acc:
                best_valid_acc = valid_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            print(f'Epoch {epoch:2d}; train loss {train_loss:.4f}, train acc {train_acc:.4f}; val acc: {valid_acc:.4f}')

            # Visualization
            model.eval()
            ret = model(test_index, test_x, test_y, epoch)
            view_e, fusion_e, loss, loss_c, loss_d = (ret[i] for i in ['view_e', 'fusion_e', 'loss', 'loss_c', 'loss_d'])
            writer.add_scalar(f'Train_{write_time}/Accuracy', train_acc, epoch)
            writer.add_scalars(f'Train_{write_time}/Loss', {'Total Loss': loss.mean(), 'Classification Loss': loss_c.mean(), 'Degradation Loss': loss_d.mean()}, epoch)
            for v in range(model.num_views):
                writer.add_scalars(f'Train_{write_time}/U[{v}]', dict((f'({i},{j})', model.degrade_loss.U[v, i, j]) for i in range(3) for j in range(3)), epoch)
            for i in range(3):
                writer.add_scalars(f'Train_{write_time}/sample[{i}]_fusion_e(label={test_y[i]})', dict((f'({i},{j})', fusion_e[i, j]) for j in range(3)), epoch)
                for v in range(model.num_views):
                    writer.add_scalars(f'Train_{write_time}/sample[{i}]_view_e[{v}]', dict((f'({i},{j})', view_e[v][i, j]) for j in range(3)), epoch)
        writer.close()

        model.load_state_dict(best_model_wts)
        print('Validation Accuracy:', self.validate())
        if saving_path is not None:
            os.makedirs(os.path.dirname(saving_path), exist_ok=True)
            torch.save(model, saving_path)
        return model

    def validate(self, loader=None):
        if loader is None:
            loader = self.valid_loader
        self.model.eval()
        with torch.no_grad():
            correct, num_samples = 0, 0
            for batch in loader:
                ret = self.model(batch['index'], batch['x'])
                correct += torch.sum(self.model.ds_combine(ret['view_e']).cpu().argmax(dim=-1).eq(batch['y'])).item()
                num_samples += len(batch['y'])
        acc = correct / num_samples
        return acc


if __name__ == '__main__':
    Experiment().train()
