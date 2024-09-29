import os
from utils.m4_summary import M4Summary
import json
import argparse




parser = argparse.ArgumentParser(description='GPT4TS')
parser.add_argument('--model', type=str, default='GPT4TS_multi')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')

args = parser.parse_args()

folder_path = './m4_results/' + args.model + '-' + args.model_comment + '/'
file_path = folder_path

if 'Weekly_forecast.csv' in os.listdir(file_path) \
        and 'Monthly_forecast.csv' in os.listdir(file_path) \
        and 'Yearly_forecast.csv' in os.listdir(file_path) \
        and 'Daily_forecast.csv' in os.listdir(file_path) \
        and 'Hourly_forecast.csv' in os.listdir(file_path) \
        and 'Quarterly_forecast.csv' in os.listdir(file_path):
    m4_summary = M4Summary(file_path, args.root_path)
    # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
    smape_results, owa_results, mape, mase = m4_summary.evaluate()
    print('smape:', smape_results)
    print('mape:', mape)
    print('mase:', mase)
    print('owa:', owa_results)
    print(f"dumping to {os.path.join(file_path, 'results.json')}")
    with open(os.path.join(file_path, 'results.json'), 'w+') as f:
        json.dump({
            'smape': smape_results,
            'mape': mape,
            'mase': mase,
            'owa': owa_results
        }, f)
else:
    print('After all 6 tasks are finished, you can calculate the averaged performance')