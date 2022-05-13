
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
import optuna as opt
import torch
import os
import joblib


# space for hyper parameters declaration

def make_save_cv_model(i,model_name,model,best_params,optim,output_path="../outputs/cross_validated_models"):

    ''' This function saves cross validation model in the corresponding directory ( if the path does not exist it creates the path for it'''


    if os.path.exists(os.path.join(f"{output_path}/{i}_{model_name}_{optim}")):
        joblib.dump(model, os.path.join(f"{output_path}/{i}_{model_name}_{optim}/{i}_model.z"))
        with open(os.path.join(f"{output_path}/{i}_{model_name}_{optim}/model_params.txt"),"w+") as file:
            file.write(best_params)
    else:
        os.mkdir(os.path.join(f"{output_path}/{i}_{model_name}_{optim}"))
        joblib.dump(model, os.path.join(f"{output_path}/{i}_{model_name}_{optim}/{i}_model.z"))
        with open(os.path.join(f"{output_path}/{i}_{model_name}_{optim}/model_params.txt"),"w+") as file:
            file.write(best_params)

def train(model_name,sc_df,tar_df,optim,k_folds=10,tar_cols="", exclude_cols =[], verbose=1):

    ''' this function is used to train the model with parameters optimization using optuna and cross validation using stratified k_folds'''

    print("[++] Starting the training process ...")
    droper = exclude_cols
    droper.append(tar_cols)
    x = sc_df.drop(droper, axis=1)
    index_list = []
    for col in range(len(x.columns)):
        index = 0
        for i in x.iloc[:,col]:
            if np.isnan(i):
                index_list.append(index)
            if not np.isfinite(i):
                index_list.append(index)
            index += 1
    print(list(set(index_list)))
    print(len(list(set(index_list))))
    y = tar_df[tar_cols]
    # k_fold constructing the cross-validation framework
    skf = StratifiedKFold(n_splits=k_folds,shuffle=True, random_state=123 )
    model_name = model_name 
    for i, (train_index, test_index) in enumerate(skf.split(x,y)):   
        def objective(trial):
            clf = TabNetClassifier(n_d=trial.suggest_int("n_d", 8, 64),
                                    n_a =trial.suggest_int("n_a", 8, 64),
                                    n_steps = trial.suggest_int("n_steps",3,10),
                                    gamma =trial.suggest_float("gamma", 1.0, 2.0),
                                    n_independent = trial.suggest_int("n_independent",1,5),
                                    n_shared = trial.suggest_int("n_shared",1,5),
                                    momentum = trial.suggest_float("momentum", 0.01, 0.4),
                                    optimizer_fn = torch.optim.Adam,
                                    optimizer_params = dict(lr=trial.suggest_float("lr",1e-4,1e-3)),
                                    scheduler_fn = torch.optim.lr_scheduler,
                                    scheduler_params = {"gamma" :trial.suggest_float("sch-gamma", 0.5, 0.95), "step_size": trial.suggest_int("sch_step_size", 10, 20, 2)},
                                    verbose = verbose,
                                    device_name = "auto"
                                    )
            X_train,X_test = x.iloc[train_index], x.iloc[test_index]
            Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
            print(list(set(index_list)))
            print(len(list(set(index_list))))
            clf.fit(X_train, Y_train,
                    eval_set = [(X_test, Y_test)])
            Y_pred = clf.predict(X_test)
            acc = accuracy_score(Y_pred, Y_test)
            return acc

        print(f"Starting optimization for fold : [{i}/{k_folds}]")
        study = opt.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        print(f" Best params for fold : [{i}/{k_folds}]")
        print(best_params)

        clf_model = TabNetClassifier(best_params)
        try:
            print("[++] Saved the model and parameters in corresponding directories")
            make_save_cv_model(i,model_name,clf_model,best_params,optim=optim)
        except:
            print("[-] Failed to save the model")
        print("[++] Ended the training process ...")





if __name__ == '__main__':
    use_df = pd.read_csv("../inputs/standard_ml_preprocessed_df.csv")
    # for key,item in use_df.isna().sum():
    #     print(f"{key} = {item}"
    # print(tar_df.isna().sum())
    tar_df = pd.read_csv("../inputs/unscaled_preprocessed_df.csv")
    tar_col = "PCE_categorical"

    exclude_cols = ["JV_default_PCE_numeric","JV_average_over_n_number_of_cells_numeric"]
    model_name = "pytorch_tabnet"
    optimizer = "Adam"
    folds = 6
    train(model_name=model_name,
        sc_df=use_df,
        tar_df=tar_df,
        tar_cols=tar_col,
        exclude_cols=exclude_cols,
        optim=optimizer,
        k_folds=folds)
