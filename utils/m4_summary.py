# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright 2020 Element AI Inc. All rights reserved.

"""
M4 Summary
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from data_provider.m4 import M4Dataset
from data_provider.m4 import M4Meta
import os

from scipy.stats import ttest_1samp
from scipy import stats

def group_values(values, groups, group_name):
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])


def mase(forecast, insample, outsample, frequency):
    return np.mean(np.abs(forecast - outsample)) / np.mean(np.abs(insample[:-frequency] - insample[frequency:]))


def smape_2(forecast, target):
    denom = np.abs(target) + np.abs(forecast)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 200 * np.abs(forecast - target) / denom


def mape(forecast, target):
    denom = np.abs(target)
    # divide by 1.0 instead of 0.0, in case when denom is zero the enumerator will be 0.0 anyway.
    denom[denom == 0.0] = 1.0
    return 100 * np.abs(forecast - target) / denom


class M4Summary:
    def __init__(self, file_path, root_path):
        self.file_path = file_path
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv')

    def evaluate(self):
        """
        Evaluate forecasts using M4 test dataset.

        :param forecast: Forecasts. Shape: timeseries, time.
        :return: sMAPE and OWA grouped by seasonal patterns.
        """
        grouped_owa = OrderedDict()

        naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)
        naive2_forecasts = np.array([v[~np.isnan(v)] for v in naive2_forecasts])

        model_mases = {}
        naive2_smapes = {}
        naive2_mases = {}
        grouped_smapes = {}
        grouped_mapes = {}
        
        # Individual values (new)
        model_mases_individual = {}
        naive2_smapes_individual = {}
        naive2_mases_individual = {}
        grouped_smapes_individual = {}
        grouped_mapes_individual = {}
        for group_name in M4Meta.seasonal_patterns:
            file_name = self.file_path + group_name + "_forecast.csv"
            if os.path.exists(file_name):
                model_forecast = pd.read_csv(file_name).values

            naive2_forecast = group_values(naive2_forecasts, self.test_set.groups, group_name)
            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            # all timeseries within group have same frequency
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)
            # Calculate individual MASE values
            model_mases_individual[group_name] = [
                mase(forecast=model_forecast[i],
                    insample=insample[i],
                    outsample=target[i],
                    frequency=frequency) for i in range(len(model_forecast))
            ]
            naive2_mases_individual[group_name] = [
                mase(forecast=naive2_forecast[i],
                    insample=insample[i],
                    outsample=target[i],
                    frequency=frequency) for i in range(len(model_forecast))
            ]
            
            model_mases[group_name] = np.mean([mase(forecast=model_forecast[i],
                                                    insample=insample[i],
                                                    outsample=target[i],
                                                    frequency=frequency) for i in range(len(model_forecast))])
            naive2_mases[group_name] = np.mean([mase(forecast=naive2_forecast[i],
                                                     insample=insample[i],
                                                     outsample=target[i],
                                                     frequency=frequency) for i in range(len(model_forecast))])
            # Calculate individual SMAPE and MAPE values
            naive2_smapes_individual[group_name] = smape_2(naive2_forecast, target)
            grouped_smapes_individual[group_name] = smape_2(forecast=model_forecast, target=target)
            grouped_mapes_individual[group_name] = mape(forecast=model_forecast, target=target)
            
            naive2_smapes[group_name] = np.mean(smape_2(naive2_forecast, target))
            grouped_smapes[group_name] = np.mean(smape_2(forecast=model_forecast, target=target))
            grouped_mapes[group_name] = np.mean(mape(forecast=model_forecast, target=target))

        results = bootstrap_owa_significance(grouped_smapes_individual, grouped_mapes_individual, model_mases_individual, 
                             naive2_smapes_individual, naive2_mases_individual, n_iterations=1000, 
                             confidence_level=0.95)
        print(results)
        
        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
        grouped_naive2_mases = self.summarize_groups(naive2_mases)
        for k in grouped_model_mases.keys():
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                              grouped_smapes[k] / grouped_naive2_smapes[k]) / 2    
    
        def round_all(d):
            return dict(map(lambda kv: (kv[0], np.round(kv[1], 3)), d.items()))

        return round_all(grouped_smapes), round_all(grouped_owa), round_all(grouped_mapes), round_all(
            grouped_model_mases)

    def summarize_groups(self, scores):
        """
        Re-group scores respecting M4 rules.
        :param scores: Scores per group.
        :return: Grouped scores.
        """
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len(np.where(self.test_set.groups == group_name)[0])

        weighted_score = {}
        for g in ['Yearly', 'Quarterly', 'Monthly']:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            scores_summary[g] = scores[g]
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score['Others'] = others_score
        scores_summary['Others'] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary['Average'] = average

        return scores_summary


def bootstrap_owa_significance(grouped_smapes, grouped_mapes, model_mases, 
                             naive2_smapes, naive2_mases, n_iterations=1000, 
                             confidence_level=0.95):
    """
    Perform bootstrap significance testing for OWA scores.
    
    Parameters:
    -----------
    grouped_smapes, grouped_mapes, model_mases, naive2_smapes, naive2_mases : dict
        Dictionaries containing grouped metrics where keys are group identifiers
        and values are arrays of metrics
    n_iterations : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level for intervals (default: 0.95)
    
    Returns:
    --------
    dict : Contains bootstrap statistics for each group including:
           - original_owa: Original OWA score
           - ci_lower: Lower confidence interval
           - ci_upper: Upper confidence interval
           - p_value: p-value for null hypothesis (OWA = 0.5)
    """
    results = {}
    
    smape_dict = {
        'Hourly': 25.942,
        'Daily': 3.606,
        'Weekly': 1.905,
        'Monthly': 12.975,
        'Quarterly': 10.319,
        'Yearly': 13.466
    }

    mase_dict = {
        'Hourly': 7.532,
        'Daily': 1.097,
        'Weekly': 3.97,
        'Monthly': 0.96,
        'Quarterly': 3.216,
        'Yearly': 3.036
    }

    owa_dict = {
        'Hourly': 2.278,
        'Daily': 1.202,
        'Weekly': 1.365,
        'Monthly': 0.901,
        'Quarterly': 0.909,
        'Yearly': 0.794
    }

    # print(grouped_smapes)
    for k in grouped_smapes.keys():
        # Get arrays for current group
        smapes = np.array(grouped_smapes[k])
        mapes = np.array(grouped_mapes[k])
        model_mas = np.array(model_mases[k])
        naive2_smape = np.array(naive2_smapes[k])
        naive2_mas = np.array(naive2_mases[k])
        
        # Calculate original OWA
        original_owa = (model_mas.mean() / naive2_mas.mean() + 
                       smapes.mean() / naive2_smape.mean()) / 2
        
        # Storage for bootstrap samples
        bootstrap_owas = np.zeros(n_iterations)
        # print(type(smapes))
        # print(smapes.shape)
        # Get sample size
        n_samples = len(smapes)
        
        # Perform bootstrap iterations
        for i in range(n_iterations):
            # Generate bootstrap indices
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Calculate OWA for bootstrap sample
            bootstrap_owas[i] = (
                model_mas[indices].mean() / naive2_mas[indices].mean() +
                smapes[indices].mean() / naive2_smape[indices].mean()
            ) / 2
        
        # Calculate confidence intervals
        ci_lower, ci_upper = np.percentile(bootstrap_owas, 
                                         [(1 - confidence_level) * 100 / 2, 
                                          (1 + confidence_level) * 100 / 2])
        
        # Calculate p-value (two-tailed test against null hypothesis OWA = 0.5)
        # We use the bootstrap distribution to estimate the p-value
        null_value_owa = owa_dict[k]
        t_stat = (original_owa - null_value_owa) / np.std(bootstrap_owas)
        print(t_stat)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        res_mase = stats.ttest_1samp(model_mas.reshape(-1), popmean=mase_dict[k])
        print(res_mase)
        res_smape = stats.ttest_1samp(smapes.reshape(-1), popmean=smape_dict[k])
        print(res_smape.pvalue)
        print(smapes.shape)
        results[k] = {
            'original_owa': original_owa,
            'reference_owa': null_value_owa,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value_owa': p_value,
            'original_smape': np.mean(smapes),
            'reference_smape': smape_dict[k],
            'p_value_smape': res_smape.pvalue,
            'original_mase': np.mean(model_mas),
            'reference_mase': mase_dict[k],
            'p_value_mase': res_mase.pvalue,
            # 'bootstrap_samples': bootstrap_owas
        }
    
    return results
