import pandas as pd
import numpy as np

datapath = 'LOO{}.dat'
train = pd.read_csv(datapath.format('Train' + '_recall'), delimiter='\t')
valid = pd.read_csv(datapath.format('Val' + '_recall'), delimiter='\t')
test = pd.read_csv(datapath.format('Test' + '_recall'), delimiter='\t')
candidates = set(range(max(train.iid.max(), valid.iid.max(), test.iid.max())))

train = train.groupby('uid').iid.apply(list)
valid = valid.groupby('uid').iid.apply(list)
test = test.groupby('uid').iid.apply(list)

history = dict()

for df in [train, valid, test]:
    for uid, iids in df.items():
        if uid not in history:
            history[uid] = set()
        history[uid] |= set(iids)

with open(datapath.format('Negatives'), 'w') as f:
    for uid, his in history.items():
        neg = np.random.choice(list(candidates - his), len(train[uid]) * 20)
        f.write(str(uid))
        for item in neg:
            f.write('\t' + str(item))
        f.write('\n')