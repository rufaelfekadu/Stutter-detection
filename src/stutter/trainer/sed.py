from .trainer import Trainer
from stutter.utils.annotation import LabelMap
from stutter.utils.data import deconstruct_labels
from stutter.utils.misc import plot_sample
from stutter.utils.metrics import compute_metrics
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class SedTrainer2(Trainer):

    def __init__(self, cfg, logger=None, metrics=['acc']):
        super(SedTrainer2, self).__init__(cfg, logger, metrics = metrics)
        self.criterion = self.criterion['t2']
        self.test_mfcc = []
        self.test_preds = []
        self.test_labels = []
        self.test_fnames = []
        self.test_encoder_outputs = []
        self.val_mfcc = []
        self.val_outputs = []
        self.val_preds = []
        
        self.num_classes = 5

        # print the number of trainable parameters

    def hook_fn(self, module, input, output):
        self.test_encoder_outputs.append(output)

    def parse_batch_train(self, batch):
        x = batch['mel_spec']
        y = batch['label']
        return x.to(self.device), y.to(self.device)
    
    def parse_batch_test(self, batch):
        x = batch['mel_spec']
        y = batch['label']
        fname = batch['file_path']
        return x.to(self.device), y.to(self.device), fname
    
    def train_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return{
            't2': loss.item()
        }
    
    def val_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        self.val_outputs.append(y)
        self.val_preds.append(pred)
        self.val_mfcc.append(x)
        return{
            't2': loss.item()
        }

    def test(self, loader=None, name='test'):
        # generate the reference txt_files
        loader = loader or self.test_loader
        self.stage = 'test'
        self.model.eval()
        self._reset_meters(self.test_meters)
        handle = self.model.encoder.register_forward_hook(self.hook_fn)
        with torch.no_grad():
            for batch in loader:

                x, y, fname = self.parse_batch_test(batch)
                preds = self.model(x.squeeze(1).squeeze(1), output_attentions=True)
                test_loss = self.criterion(preds, y)
                self.test_mfcc.append(x)
                self.test_preds.append(preds)
                self.test_fnames.append(fname)
                self.test_labels.append(y)
                self._update_meter(self.test_meters, f't2_test_loss', test_loss.item())
        
        self._write_meters(self.test_meters)
        handle.remove()
        self.after_test()

    def after_test(self):
        mfcc = torch.cat(self.test_mfcc, axis=0)
        preds = torch.concat(self.test_preds, axis=0) # (N, 22, 2+num_classes)
        fnames = self.test_fnames[0]
        y = torch.cat(self.test_labels, axis=0)
        preds_mask = (torch.sigmoid(preds) >= 0.5).int()

        # write the predictions to a file
        for i,fname in tqdm(enumerate(fnames), desc='Writing predictions', total=len(fnames)):
            pred_fname = fname.replace('_ref.txt', '_pred.txt')
            pred = preds_mask[i,:,:]
            with open(pred_fname, 'w') as f:
                events = deconstruct_labels(pred, clip_duration=30, sr=16000, smooth=3)
                for event in events:
                    f.write(f'{event[0]},{event[1]},{event[2]}\n')
        
        # plot some of the encoder outputs
        output = torch.concat(self.test_encoder_outputs, axis=0)
        preds = torch.nn.functional.interpolate(preds.unsqueeze(0), size=(output.size(1), output.size(2)), mode='nearest').squeeze(0)
        preds_mask = torch.nn.functional.interpolate(preds_mask.float().unsqueeze(0), size=(output.size(1), output.size(2)), mode='nearest').squeeze(0)
        resized_y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(output.size(1), output.size(2)), mode='nearest').squeeze(0)
        for i in range(5):
            fig, ax = plt.subplots(5, figsize=(20,10))
            normalized_output = (output[i] - output[i].min()) / (output[i].max() - output[i].min())
            # normalized_mfcc = (mfcc[i] - mfcc[i].min()) / (mfcc[i].max() - mfcc[i].min())

            # ax[0].imshow(normalized_mfcc.cpu().numpy().T, aspect='auto', cmap='inferno')
            # ax[0].set_title('MFCC Features')

            ax[1].imshow(output[i].cpu().numpy().T, aspect='auto', cmap='inferno')
            ax[1].set_title('wav2vec2 Features')

            ax[2].imshow(preds[i].cpu().numpy().T, aspect='auto', cmap='inferno')
            ax[2].set_title('logits')
        
            ax[3].imshow(preds_mask[i].cpu().numpy().T, aspect='auto', cmap='inferno')
            ax[3].set_title('Predictions')

            ax[4].imshow(resized_y[i].cpu().numpy().T, aspect='auto', cmap='inferno')
            ax[4].set_title('Ground Truth')
            
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])

            # Add a single color bar
            cbar = fig.colorbar(ax[1].images[0], ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            plt.savefig(f'outputs/encoder_output_{i}.png')
            self.logger.add_figure(f'test/encoder_output_{i}', fig)
                
class SedTrainer(Trainer):
    
    def __init__(self, cfg, logger=None, metrics=['acc']):
        super(SedTrainer, self).__init__(cfg, logger, metrics = metrics)
        self.criterion = self.criterion['t2']
        self.test_preds = []
        self.test_labels = []
        self.test_fnames = []
        self.num_classes = 5

    def parse_batch_train(self, batch):
        x = batch['audio']
        y = batch['label']
        return x.to(self.device), y.to(self.device)
    
    def parse_batch_test(self, batch):
        x = batch['audio']
        y = batch['label']
        return x.to(self.device), y.to(self.device)
    
    def train_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.unsqueeze(1))
        loss = self.criterion(pred, y)
        loss.backward()
        self.optimizer.step()

        return{
            't2': loss.item()
        }
    
    def val_step(self, batch):
        x, y = self.parse_batch_train(batch)
        pred = self.model(x.squeeze(1))
        loss = self.criterion(pred, y)
        metrics = compute_metrics(pred, y, self.num_classes)
        return{
            't2': loss.item(),
            **metrics
        }
    
    def test(self, loader=None, name='test'):
        # generate the reference txt_files
        loader = loader or self.test_loader
        self.stage = 'test'
        self.model.eval()
        self._reset_meters(self.test_meters)

        with torch.no_grad():
            for batch in loader:
                x, y = self.parse_batch_test(batch)
                preds = self.model(x.squeeze(1))
                test_loss = self.criterion(preds, y)
                self.test_preds.append(preds)
                self._update_meter(self.test_meters, f't2_test_loss', test_loss.item())
        
        self._write_meters(self.test_meters)
        self.after_test()

    def after_test(self):
        label_map = LabelMap()
        preds = torch.concat(self.test_preds, axis=0) # (N, 22, 2+num_classes)
        fnames = self.test_fnames[0]
        preds[:,:,2:] = (torch.sigmoid(preds[:,:,2:]) >= 0.5).int()
        # write the predictions to a file
        for i,fname in enumerate(fnames):
            pred_fname = fname.replace('.txt', '_pred.txt')
            pred = preds[i,:,:]
            with open(pred_fname, 'w') as f:
                for j in range(pred.size(0)):
                    for k, l in enumerate(pred[j,2:]):
                        if l == 1 and pred[j,1] > 0:
                            start_l = pred[j,0].item()
                            end_l = pred[j,1].item()
                            f.write(f'{round(start_l,2)},{round(end_l,2)},{label_map.description[label_map.core[k]]}\n')
             