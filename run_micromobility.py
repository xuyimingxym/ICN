import os
import torch
from datetime import datetime
from experiments.exp_micromobility import Exp_Micromobility
import argparse
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='ICN on shared micromobility datasets')

### -------  dataset settings --------------
parser.add_argument('--dataset', type=str, default='Austin', choices=['Austin', 'Chicago', 'Chicago_Halfyear'])  
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--normtype', type=int, default=2)

### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--device', type=str, default='cuda:0')

### -------  input/output length settings --------------                                                                            
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--concat_len', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)

parser.add_argument('--train_length', type=float, default=6)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=2)

### -------  training settings --------------  
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--tune', type=bool, default=False)
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--ablation', type=str, default=None, choices=[None, 'poi', 'demo', 'tran'])
parser.add_argument('--weather', type=bool, default=False)

parser.add_argument('--epoch', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--optimizer', type=str, default='RMSProp') #
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)

parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='ICN')

### -------  model settings --------------  
parser.add_argument('--hidden_size', default=0.0625, type=float, help='hidden channel scale of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size for the first layer')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--positionalEcoding', type=bool , default = True)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=2)
parser.add_argument('--stacks', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_decoder_layer', type=int, default=1)
parser.add_argument('--RIN', type=bool, default=False)


args = parser.parse_args()

if __name__ == '__main__':

    torch.manual_seed(4321)  # reproducible. 4321 is a magical seed lol.
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    Exp=Exp_Micromobility
    # exp=Exp(args)

    if args.evaluate:
        exp=Exp(args)
        before_evaluation = datetime.now().timestamp()
        exp.test()
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    elif args.train or args.resume:
        exp=Exp(args)
        before_train = datetime.now().timestamp()
        print("===================Normal-Start=========================")
        best_epoch_val, best_validate_metrics, best_epoch_test, best_test_metrics = exp.train()
        after_train = datetime.now().timestamp()
        # print best epoch
        print('Best validation epoch:', best_epoch_val)
        print('Best test epoch:', best_epoch_test)
        print('Performance on test set: | MAE: {:5.4f} | MAPE: {:5.4f} | RMSE: {:5.4f}'.format(best_test_metrics['mae'],
                                                                                               best_test_metrics['mape'],
                                                                                               best_test_metrics['rmse']))
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Normal-End=========================")

    elif args.tune:
        before_tune = datetime.now().timestamp()

        horizon_list = [1, 6, 12]
        window_list = [24, 48, 96, 192]
        batch_list = [8, 16, 32, 64]
        hidden_list = [0.5, 1]
        # lr_list = [0.001, 0.0005]
        # horizon_list = [12]
        # window_list = [96, 192]
        # batch_list = [8, 16, 32, 64]
        # hidden_list = [0.5, 1]
        lr_list = [0.001]

        results = pd.DataFrame(columns=['Horizon', 'Window', 'Batch_Size', 'Hidden_Size', 
                                        'Initial_Lr', 'MAE', 'MAPE', 'RMSE'])

        print("===================Tuning-Start=========================")

        for horizon in horizon_list:
            for window in window_list:
                for batch in batch_list:
                    for hidden in hidden_list:
                        for lr in lr_list:
                            args.horizon = horizon
                            args.window_size = window
                            args.batch_size = batch
                            args.hidden_size = hidden
                            args.lr = lr
                            exp=Exp(args)
                            best_epoch_val, best_validate_metrics, best_epoch_test, best_test_metrics = exp.train()
                            best_case = pd.DataFrame([[horizon, window, batch, hidden, lr, 
                                                       best_test_metrics['mae'], best_test_metrics['mape'],
                                                       best_test_metrics['rmse']]], columns=results.columns)
                            results = pd.concat([results, best_case])
                            with open(args.dataset + '_Tuning.txt', 'a') as f:
                                f.write(str(horizon) + ',' + str(window) + ',' + str(batch) + ',' +
                                        str(hidden) + ',' + str(lr) + ',' + str(best_test_metrics['mae']) + 
                                        ',' + str(best_test_metrics['mape']) + ',' + str(best_test_metrics['rmse'])
                                        + '\n')

        after_tune = datetime.now().timestamp()
        results.to_csv('Results/' + args.dataset + '_Tuning.csv')
        print(f'Tuning took {(after_tune - before_tune) / 60} minutes')
        print("===================Tuning-End=========================")



