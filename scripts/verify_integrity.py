'Local NLP project module: verify integrity.\n\nSee docs/CODE_WALKTHROUGH.md for dependencies and architecture.'
from typing import Any as _DocAny
import pandas as pd
import json

df_train = pd.read_csv('data/processed/ragtruth_claims_train.csv')
df_val = pd.read_csv('data/processed/ragtruth_claims_val.csv')
df_test = pd.read_csv('data/processed/ragtruth_claims_test.csv')
with open('data/processed/split_manifest.json') as f:
    manifest = json.load(f)

# 1. Source overlap
train_src = set(df_train['source_id'])
val_src = set(df_val['source_id'])
test_src = set(df_test['source_id'])

tv = len(train_src & val_src)
tt = len(train_src & test_src)
vt = len(val_src & test_src)
print(f'Source Overlap: Train-Val={tv}, Train-Test={tt}, Val-Test={vt}')
assert tv == 0 and tt == 0 and vt == 0, 'Data leakage detected!'

# 2. Counts matching files
print(f"Train claims: {len(df_train)} (Manifest: {manifest['train_claims_total']})")
print(f"Val claims: {len(df_val)} (Manifest: {manifest['val_claims_total']})")
print(f"Test claims: {len(df_test)} (Manifest: {manifest['test_claims_total']})")
assert len(df_train) == manifest['train_claims_total']
assert len(df_val) == manifest['val_claims_total']
assert len(df_test) == manifest['test_claims_total']

# 3. Claims not null and labels in {0, 1}
assert df_train['label'].isin([0, 1]).all()
assert df_val['label'].isin([0, 1]).all()
assert df_test['label'].isin([0, 1]).all()
print('All labels valid in {0, 1}')

# 4. Check claim text not empty
assert (df_train['claim'].str.len() > 0).all()
assert (df_val['claim'].str.len() > 0).all()
assert (df_test['claim'].str.len() > 0).all()
print('All claims are non-empty text.')

print('DATA INTEGRITY VERIFICATION PASSED!')
