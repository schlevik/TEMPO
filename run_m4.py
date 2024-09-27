import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler

from data_provider.m4 import M4Meta

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import pandas

from models.DLinear import DLinear
from models.ETSformer import ETSformer
from models.GPT4TS import GPT4TS
from models.PatchTST import PatchTST
from models.T5 import T54TS
from models.TEMPO import TEMPO
from utils.losses import smape_loss
from utils.m4_summary import M4Summary
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStoppingM4, adjust_learning_rate_m4, load_content, test_m4

# import builtins
# from inspect import getframeinfo, stack
# original_print = print

# def print_wrap(*args, **kwargs):
#     caller = getframeinfo(stack()[1][0])
#     original_print("FN:",caller.filename,"Line:", caller.lineno,"Func:", caller.function,":::", *args, **kwargs)

# builtins.print = print_wrap

test = test_m4
adjust_learning_rate = adjust_learning_rate_m4
EarlyStopping = EarlyStoppingM4

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, default='weather_GTP4TS_multi-debug')


parser.add_argument('--prompt', type=int, default=0)
parser.add_argument('--num_nodes', type=int, default=1)


parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.9)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type3') # for what
parser.add_argument('--patience', type=int, default=20)

parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=768)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='GPT4TS_multi')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--equal', type=int, default=1, help='1: equal sampling, 0: dont do the equal sampling')
parser.add_argument('--pool', action='store_true', help='whether use prompt pool')
parser.add_argument('--no_stl_loss', action='store_true', help='whether use prompt pool')

parser.add_argument('--stl_weight', type=float, default=0.01)
parser.add_argument('--config_path', type=str, default='./data_config.yml')
# parser.add_argument('--datasets', type=str, default='exchange')
# parser.add_argument('--target_data', type=str, default='ETTm1')

parser.add_argument('--use_token', type=int, default=0)
parser.add_argument('--electri_multiplier', type=int, default=1)
parser.add_argument('--traffic_multiplier', type=int, default=1)
parser.add_argument('--embed', type=str, default='timeF')


# text prototype
parser.add_argument('--type_of_prototype', type=str, default='representative',
                    help='Provide or select the prototypes. ([provide text list] or random or representative)')
parser.add_argument('--number_of_prototype', type=int, default=10, metavar='Number',
                    help='Number of prototype to select')

# encoder
parser.add_argument('--load_encoder', action='store_true', default=False,
                    help='activate to load the estimator instead of ' +
                            'training it')
parser.add_argument('--fit_encoder_classifier', action='store_true', default=False,
                    help='if not supervised, activate to load the ' +
                            'model and retrain the classifier')
parser.add_argument('--encoder_save_path', type=str, metavar='PATH', default='./encoders/saved_models',
                    help='path where the estimator is/should be saved')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
# parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
# parser.add_argument('--model', type=str, required=True, default='Autoformer',
#                     help='model name, options: [Autoformer, DLinear]')
# parser.add_argument('--seed', type=int, default=0, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
# parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
# parser.add_argument('--label_len', type=int, default=48, help='start token length')
# parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# # model define
# parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=7, help='output size')
# parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# parser.add_argument('--factor', type=int, default=1, help='attn factor')
# parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
# parser.add_argument('--embed', type=str, default='timeF',
#                     help='time features encoding, options:[timeF, fixed, learned]')
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
# parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
# parser.add_argument('--stride', type=int, default=8, help='stride')
# parser.add_argument('--prompt_domain', type=int, default=0, help='')
# parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
# parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
# parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
# parser.add_argument('--itr', type=int, default=1, help='experiments times')
# parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
# parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
# parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

folder_path = './m4_results/' + args.model + '-' + args.model_comment + '/'
file_path = folder_path
for ii in range(args.itr):
    # setting record of experiments
    if args.is_training:
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            1,
            args.d_ff,
            1,
            args.embed,
            args.des, ii)

        if args.data == 'm4':
            args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]  # Up to M4 config
            args.seq_len = 2 * args.pred_len
            args.label_len = args.pred_len
            args.frequency_map = M4Meta.frequency_map[args.seasonal_patterns]

        # config = setup(args)
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        accelerator.print("Creating LLM model ...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        elif args.model == 'TEMPO':
            model = TEMPO(args, device)
            model.to(device)
        elif args.model == 'T5':
            model = T54TS(args, device)
            model.to(device)
        elif 'ETSformer' in args.model:
            model = ETSformer(args, device)
            model.to(device)
        else:
            model = GPT4TS(args, device)
        
        # model=model.to(device=device)

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, verbose=True)

        model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = smape_loss()

        train_loader, vali_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, model, model_optim, scheduler)

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, trend, season, resid) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(accelerator.device)

                batch_y = batch_y.float().to(accelerator.device)
                batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                # dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                #     accelerator.device)

                outputs, loss_local = model(batch_x, ii, trend, season, resid)

                f_dim = -1 if args.features == 'MS' else 0
                # print(f_dim)
                # print(args.pred_len)
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                batch_y_mark = batch_y_mark[:, -args.pred_len:, f_dim:]
                loss = criterion(batch_x, args.frequency_map, outputs, batch_y, batch_y_mark)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                accelerator.backward(loss)
                model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = test(args, accelerator, model, train_loader, vali_loader, criterion)
            test_loss = vali_loss
            accelerator.print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, model, path)  # model saving
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args)
            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint'
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        unwrapped_model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))

        x, _, trend, season, noise  = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
        x = x.unsqueeze(-1)
        
        trend = torch.tensor(trend, dtype=torch.float32).to(accelerator.device)
        trend = trend.unsqueeze(-1)
        
        season = torch.tensor(season, dtype=torch.float32).to(accelerator.device)
        season = season.unsqueeze(-1)
        
        noise = torch.tensor(noise, dtype=torch.float32).to(accelerator.device)
        noise = noise.unsqueeze(-1)

        model.eval()

        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
            dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
            outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
            id_list = np.arange(0, B, args.eval_batch_size)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                out, _ = model(
                x[id_list[i]:id_list[i + 1]], -1, trend[id_list[i]:id_list[i + 1]], season[id_list[i]:id_list[i + 1]], noise[id_list[i]:id_list[i + 1]]
                )
                outputs[id_list[i]:id_list[i + 1], :, :] = out
            accelerator.wait_for_everyone()
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

        accelerator.print('test shape:', preds.shape)
        
        
        if not os.path.exists(folder_path) and accelerator.is_local_main_process:
            os.makedirs(folder_path)

        if accelerator.is_local_main_process:
            forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(args.pred_len)])
            forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
            forecasts_df.index.name = 'id'
            forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
            forecasts_df.to_csv(folder_path + args.seasonal_patterns + '_forecast.csv')

            # calculate metrics
            accelerator.print(args.model)
            
            
    if accelerator.is_local_main_process:
        if 'Weekly_forecast.csv' in os.listdir(file_path) \
                and 'Monthly_forecast.csv' in os.listdir(file_path) \
                and 'Yearly_forecast.csv' in os.listdir(file_path) \
                and 'Daily_forecast.csv' in os.listdir(file_path) \
                and 'Hourly_forecast.csv' in os.listdir(file_path) \
                and 'Quarterly_forecast.csv' in os.listdir(file_path):
            m4_summary = M4Summary(file_path, args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            accelerator.print('smape:', smape_results)
            accelerator.print('mape:', mape)
            accelerator.print('mase:', mase)
            accelerator.print('owa:', owa_results)
        else:
            accelerator.print('After all 6 tasks are finished, you can calculate the averaged performance')

accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    path = './checkpoints'  # unique checkpoint saving path
    del_files(path)  # delete checkpoint files
    accelerator.print('success delete checkpoints')

