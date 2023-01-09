import numpy as np
# Copyright © 2005-2019, NumPy Developers.

import pandas as pd
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team

def add_tailing_zeros_decimals(num,num_decimals):
    # Since the rounding of numpy exclude tailing zeros
    while len(num[num.rfind('.')+1:])!=num_decimals:
        num = num+'0'
    return num

def thousand_seperator(number):
    # Formatting numbers for Excel file
    return "{:,.0f}".format(number)