#!/usr/bin/env python3
import pandas as pd

csv_path = '/nrs/cellmap/ackermand/cellmap/analysisResults/c-elegans/jrc_c-elegans-op50-1/mito_filled_with_skeleton.csv'

df = pd.read_csv(csv_path, nrows=3)
print('Original columns:', list(df.columns))
print()

df.columns = df.columns.str.lower()
print('After lowercase:', list(df.columns))
print()

# Check for object id column
if 'object id' in df.columns:
    print('Found "object id" column')
    print(df[['object id']].head(3))
elif 'object_id' in df.columns:
    print('Found "object_id" column')
    print(df[['object_id']].head(3))
else:
    print('No object_id column found!')
    print('Available:', list(df.columns))
