import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy.io
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_df(df, user_col, exclude=[]):
    inside = df[~df.iloc[:, user_col].isin(exclude)]
    outside = df[df.iloc[:, user_col].isin(exclude)]
    return inside, outside

def drop_cols(df, cols=[]):
    return df.drop(columns=cols)

def to_tensor(df, class_col, scale=False, scaler=None):
    # data = {
    #     'data': [
    #         [
    #             768_dimension_of_ViT_embedding,
    #             the_label
    #         ],
    #         [],
    #         ...
    #     ],
    #     'targets': labels/classes as a whole
    # }
    X = df.iloc[:, :class_col].to_numpy()
    y = df.iloc[:, class_col].to_numpy()

    if scaler:
        X = scaler.transform(X)
    if scale:
        sc = StandardScaler()
        X = sc.fit_transform(X)

    y_map = {v: k for k, v in enumerate(np.unique(y))}
    y = torch.tensor(np.vectorize(y_map.get)(y)).type(torch.int)
    data = [[torch.tensor(x_, dtype=torch.float), y_] for x_, y_ in zip(X, y)]
    
    if scale:
        return {'data': data, 'targets': y, 'scaler': sc}
    return {'data': data, 'targets': y}

def make(df, user_col, exclude, drop=None, scale=False):
    inside, outside = split_df(df, user_col, exclude)
    inside = drop_cols(inside, [user_col])
    outside = drop_cols(outside, [user_col])

    if drop:
        inside = drop_cols(inside, drop)
        outside = drop_cols(outside, drop)

    if scale:
        train_data = to_tensor(inside, len(inside.columns)-1, scale)
        test_data = to_tensor(outside, len(inside.columns)-1, scaler=train_data['scaler'])
        del train_data['scaler']
    else:
        train_data = to_tensor(inside, len(inside.columns)-1)
        test_data = to_tensor(outside, len(inside.columns)-1)

    return train_data, test_data

def split_agents(df, agent_col, class_col):
    agents = df.iloc[:, agent_col].unique()
    classes = df.iloc[:, class_col].unique()

    test_agent = np.random.choice(agents, 2, replace=False)
    train_agent = np.setdiff1d(agents, test_agent)
    inside, outside = split_df(df, agent_col, test_agent)
    
    percentage = outside.shape[0]/df.shape[0]
    all_equal = np.array_equal(outside.iloc[:, class_col].unique(), classes)

    return percentage, all_equal, test_agent, train_agent

def make_dsads(path):
    dsads = scipy.io.loadmat(path)
    dsads_df = pd.DataFrame(dsads['data_dsads'])
    
    agents = dsads_df.iloc[:, 407].unique()
    test_agent = np.random.choice(agents, 2, replace=False)
    train_agent = np.setdiff1d(agents, test_agent)
    # print(train_agent, test_agent)
    train_set, test_set = make(dsads_df, user_col=407, exclude=test_agent.tolist(), drop=405)
    # print(len(train_set["targets"]), len(test_set["targets"]))
    return train_set, test_set

def make_pamap(path):
    pamap = scipy.io.loadmat(path)
    pamap_df = pd.DataFrame(pamap['data_pamap'])

    percentage, all_equal, test_agent, train_agent = split_agents(pamap_df, 244, 243)

    while percentage < 0.20 or percentage > 0.26 or not all_equal:
        percentage, all_equal, test_agent, train_agent = split_agents(pamap_df, 244, 243)

    train_set, test_set = make(pamap_df, user_col=244, exclude=test_agent.tolist())
    return train_set, test_set

def make_wisdm(path):
    # path should point to all.csv
    wisdm_df = pd.read_csv(path, header=None)
    agents = wisdm_df.iloc[:, 92].unique()
    test_agent = np.random.choice(agents, 10, replace=False)
    train_agent = np.setdiff1d(agents, test_agent)
    train_set, test_set = make(wisdm_df, user_col=92, exclude=test_agent.tolist(), scale=True)
    return train_set, test_set    

def make_hapt(path):
    train_x = np.loadtxt(f"{path}/Train/X_train.txt")
    train_y = np.loadtxt(f"{path}/Train/y_train.txt", dtype=np.int32)
    y_map = {v: k for k, v in enumerate(np.unique(train_y))}
    y = torch.tensor(np.vectorize(y_map.get)(train_y)).type(torch.int)
    data = [[torch.tensor(x_, dtype=torch.float), y_] for x_, y_ in zip(train_x, y)]
    train_data = {'data': data, 'targets': y}

    test_x = np.loadtxt(f"{path}/Test/X_test.txt")
    test_y = np.loadtxt(f"{path}/Test/y_test.txt", dtype=np.int32)
    y = torch.tensor(np.vectorize(y_map.get)(test_y)).type(torch.int)
    data = [[torch.tensor(x_, dtype=torch.float), y_] for x_, y_ in zip(test_x, y)]
    test_data = {'data': data, 'targets': y}
    return train_data, test_data

def make_flexible(path, typ='dsads'):
    if typ == 'dsads':
        new_col, final_col = 408, 405
        dsads = scipy.io.loadmat(path)
        dframe = pd.DataFrame(dsads['data_dsads'])
        dframe[new_col] = pd.Series(dtype=np.int32)
        class_ids = [i for i in range(1, 20)]
        user_ids = [i for i in range(1, 9)]
        user_col, cls_col = 407, 406
        to_drop = [405, 406, 407]        
    elif typ == 'wisdm':
        new_col, final_col = 93, 91
        dframe = pd.read_csv(path, header=None)
        dframe[new_col] = pd.Series(dtype=np.int32)
        class_ids = [i for i in range(18)]
        user_ids = [1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 
                    1613, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 
                    1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 
                    1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650]       
        user_col, cls_col = 92, 91
        to_drop = [91, 92]

    mesh_grid = np.transpose([np.tile(user_ids, len(class_ids)), np.repeat(class_ids, len(user_ids))])

    for i, m in enumerate(mesh_grid):
        # temp = dframe.loc[dframe[user_col] == m[0]]
        # if m[1] not in temp[cls_col].unique():
        #     print(f"User: {m[0]} does not have class: {m[1]}")
        dframe.loc[(dframe[cls_col] == m[1]) & (dframe[user_col] == m[0]), new_col] = i
    dframe[new_col] = dframe[new_col].astype(int)

    dframe.drop(dframe.columns[to_drop], axis=1, inplace=True)

    x_train, x_test = [], []
    for i in sorted(dframe[new_col].unique()):
        temp = dframe.loc[dframe[new_col] == i]
        x_tr, x_te = train_test_split(temp, test_size=0.2)
        x_train.append(x_tr)
        x_test.append(x_te)

    x_train = pd.concat(x_train)
    x_train.columns = [i for i in range(len(x_train.columns))]
    
    x_test = pd.concat(x_test)    
    x_test.columns = [i for i in range(len(x_test.columns))]

    # print(sorted(x_train[final_col].unique()), len(x_train[final_col].unique()))
    # print(sorted(x_test[final_col].unique()), len(x_test[final_col].unique()))
    # exit()

    if typ == 'wisdm':
        train_data = to_tensor(x_train, final_col, scale=True)
        test_data = to_tensor(x_test, final_col, scaler=train_data['scaler'])
        del train_data['scaler']
        return train_data, test_data

    return to_tensor(x_train, final_col), to_tensor(x_test, final_col)