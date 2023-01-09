import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

def get_variables_altman_1968():
    variables = [
        'EBIT / total assets',
        'retained earnings / total assets',
        'sales / total assets',
        'total equity / total liabilities',
        'working capital / total assets',
    ]
    return variables

def get_variables_altman_and_sabato_2007():
    variables = [
        'current liabilities / total equity',
        'EBITDA / interest expense',
        'EBITDA / total assets',
        'retained earnings / total assets',
        'short-term liquidity / total assets',
    ]
    return variables


def get_variables_paraschiv_2021():
    variables = [
        '(current liabilities - short-term liquidity) / total assets',
        'accounts payable / total assets',
        'dummy; one if paid-in equity is less than total equity',
        'dummy; one if total liability exceeds total assets',
        'interest expenses / total assets',
        'inventory / current assets',
        'log(age in years)',
        'net income / total assets',
        'public taxes payable / total assets',
        'short-term liquidity / current assets',
    ]
    return variables