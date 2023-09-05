import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(8) # for reproduce

path = "enter path to data"

from rdkit import Chem
import numpy as np
import sys
sys.setrecursionlimit(50000)
import pickle
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.nn.Module.dump_patches = True
import copy
import pandas as pd

from GCN import Fingerprint, save_smiles_dicts, get_smiles_array

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

ds = "clintox" # tox21, clintox, cardio
print("Dataset: ", ds)

tasks = ['label']

smiles_tasks_df = pd.read_csv(os.path.join(path, ds + '.csv'))
print("smiles_task_df.shape= ", smiles_tasks_df.shape)
smilesList = smiles_tasks_df.smiles.values
print("number of all smiles = ", len(smilesList))

atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []

for smiles in smilesList:
    try:
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    except:
        print("can not process smiles: ", smiles)
        pass

print("number of successfully processed smiles = ", len(remained_smiles))

smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
smiles_tasks_df['cano_smiles'] = canonical_smiles_list

random_seed = 888

batch_size = 32 
epochs = 150 
p_dropout = 0.5
fingerprint_dim = 1024 
 
radius = 3
T = 3 
weight_decay = 3.5 # also known as l2_regularization_lambda 
learning_rate = 3.5 # also known as alpha 
per_task_output_units_num = 2 # for classification model
output_units_num = len(tasks) * per_task_output_units_num 

feature_dicts = save_smiles_dicts(smilesList,ds)

remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)

print("uncovered_df.shape= ", uncovered_df.shape)
print("remained_df.shape= ", remained_df.shape)

weights = []
for i,task in enumerate(tasks):    
    negative_df = remained_df[remained_df[task] == 0][["smiles",task]]
    positive_df = remained_df[remained_df[task] == 1][["smiles",task]]
    weights.append([(positive_df.shape[0]+negative_df.shape[0])/negative_df.shape[0],\
                    (positive_df.shape[0]+negative_df.shape[0])/positive_df.shape[0]])
    

if ds == "cardio":
    # take the first 4001 samples for train, 4002-4157 for valid, 4158-4311 for test
    training_data = remained_df.iloc[0:4001]
    valid_data = remained_df.iloc[4001:4157]
    test_data = remained_df.iloc[4157:]
elif ds == "clintox":
    # take the first 2332 samples for train, 2333-2479 for valid, 2479-2630 for test
    training_data = remained_df.iloc[0:2332]
    valid_data = remained_df.iloc[2332:2479]
    test_data = remained_df.iloc[2479:]
else: # tox21
    # take the first 10487 samples for train, 10488-11271 for valid, 11272-12054 for test
    training_data = remained_df.iloc[0:10487]
    valid_data = remained_df.iloc[10487:11271]
    test_data = remained_df.iloc[11271:]



print("training_data.shape= ", training_data.shape)
print("valid_data.shape= ", valid_data.shape)
print("test_data.shape= ", test_data.shape)

train_df = training_data.reset_index(drop=True)
valid_df = valid_data.reset_index(drop=True)
test_df = test_data.reset_index(drop=True)

if ds == "cardio":
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[1]],feature_dicts)
elif ds == "clintox":
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[1]],feature_dicts)
else: # tox21
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[1]],feature_dicts)

num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]

loss_function = [nn.CrossEntropyLoss(torch.Tensor(weight),reduction='mean') for weight in weights]
model = Fingerprint(radius, T, num_atom_features,num_bond_features,
            fingerprint_dim, output_units_num, p_dropout)
model.cuda()

optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())


def train(model, dataset, optimizer, loss_function, epoch):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0,dataset.shape[0])
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df.cano_smiles.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        
        optimizer.zero_grad()
        loss = 0.0
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)

            loss += loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
        loss.backward()
        optimizer.step()

def eval(model, dataset):
    model.eval()
    y_val_list = {}
    y_pred_list = {}
    losses_list = []
    valList = np.arange(0,dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)   
    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch,:]
        smiles_list = batch_df.cano_smiles.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom),torch.Tensor(x_bonds),torch.cuda.LongTensor(x_atom_index),torch.cuda.LongTensor(x_bond_index),torch.Tensor(x_mask))
        atom_pred = atoms_prediction.data[:,:,1].unsqueeze(2).cpu().numpy()
        for i,task in enumerate(tasks):
            y_pred = mol_prediction[:, i * per_task_output_units_num:(i + 1) *
                                    per_task_output_units_num]
            y_val = batch_df[task].values

            validInds = np.where((y_val==0) | (y_val==1))[0]
            if len(validInds) == 0:
                continue
            y_val_adjust = np.array([y_val[v] for v in validInds]).astype(float)
            validInds = torch.cuda.LongTensor(validInds).squeeze()
            y_pred_adjust = torch.index_select(y_pred, 0, validInds)
            loss = loss_function[i](
                y_pred_adjust,
                torch.cuda.LongTensor(y_val_adjust))
            y_pred_adjust = F.softmax(y_pred_adjust,dim=-1).data.cpu().numpy()[:,1]
            losses_list.append(loss.cpu().detach().numpy())
            try:
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
            except:
                y_val_list[i] = []
                y_pred_list[i] = []
                y_val_list[i].extend(y_val_adjust)
                y_pred_list[i].extend(y_pred_adjust)
    
    eval_acc = [accuracy_score(y_val_list[i],
                               (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    eval_roc = [roc_auc_score(y_val_list[i], y_pred_list[i]) for i in range(len(tasks))]
    eval_prc = [auc(precision_recall_curve(y_val_list[i], y_pred_list[i])[1],precision_recall_curve(y_val_list[i], y_pred_list[i])[0]) for i in range(len(tasks))]
    eval_precision = [precision_score(y_val_list[i],
                                     (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    eval_recall = [recall_score(y_val_list[i],
                               (np.array(y_pred_list[i]) > 0.5).astype(int)) for i in range(len(tasks))]
    eval_loss = np.array(losses_list).mean()
    
    return eval_roc, eval_loss, eval_prc, eval_precision, eval_recall, eval_acc


best_param ={}
best_param["roc_epoch"] = 0
best_param["loss_epoch"] = 0
best_param["valid_roc"] = 0
best_param["valid_loss"] = 9e8

for epoch in range(epochs):
    train_roc, train_loss, train_prc, train_precision, train_recall, train_acc = eval(model, train_df)
    valid_roc, valid_loss, valid_prc, valid_precision, valid_recall, valid_acc = eval(model, valid_df)
   
    train_roc_mean = np.array(train_roc).mean()
    valid_roc_mean = np.array(valid_roc).mean()

    if valid_roc_mean > best_param["valid_roc"]:
        best_param["roc_epoch"] = epoch
        best_param["valid_roc"] = valid_roc_mean
        if valid_roc_mean > 0.8:
            torch.save(model, str(epoch) + '.pt')
                   
    if valid_loss < best_param["valid_loss"]:
        best_param["loss_epoch"] = epoch
        best_param["valid_loss"] = valid_loss

    print("EPOCH:\t"+str(epoch)+'\n'\
        +"train_roc_mean"+":"+str(train_roc_mean)+'\n'\
        +"valid_roc_mean"+":"+str(valid_roc_mean)+'\n'\
        )
    if (epoch - best_param["roc_epoch"] >10) and (epoch - best_param["loss_epoch"] >20):        
        break
        
    train(model, train_df, optimizer, loss_function, epoch)

# evaluate model
best_model = torch.load(str(best_param["roc_epoch"]) + '.pt')    

best_model_dict = best_model.state_dict()
best_model_wts = copy.deepcopy(best_model_dict)

model.load_state_dict(best_model_wts)
(best_model.align[0].weight == model.align[0].weight).all()
test_roc, test_loss, test_prc, test_precision, test_recall, test_acc = eval(model, test_df)

print("best epoch:"+str(best_param["roc_epoch"])
        +"\n"+"test_acc:"+str(test_acc)
        +"\n"+"test_precision:"+str(test_precision)
        +"\n"+"test_recall:"+str(test_recall)
      +"\n"+"test_roc:"+str(test_roc)
        +"\n"+"test_prc:"+str(test_prc)
      +"\n"+"test_roc_mean:",str(np.array(test_roc).mean())
        +"\n"+"test_loss:"+str(test_loss)
        
     )














