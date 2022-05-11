import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabModelClassifier
from sklearn.metrics import accuracy_score
import optuna as opt
import torch
import os
import joblib


# space for hyper parameters declaration

# space for data declaration
main_df = pd.read_csv("../inputs/standard_ml_preprocessed.csv")



def make_save_cv_model(i,model_name,model,best_params,output_path="../outputs/cross_validated_models"):
    if os.path.exists(os.path.join(f"{output_path}/{i}_{model_name}")):
        joblib.dump(model, os.path.join(f"{output_path}/{i}_{model_name}/{i}_model.z"))
        with open(os.path.join(f"{output_path}/{i}_{model_name}/model_params.txt"),"w+") as file:
            file.write(best_params)
    else:
        os.mkdir(os.path.join(f"{output_path}/{i}_{model_name}"))
        joblib.dump(model, os.path.join(f"{output_path}/{i}_{model_name}/{i}_model.z"))
        with open(os.path.join(f"{output_path}/{i}_{model_name}/model_params.txt"),"w+") as file:
            file.write(best_params)

def train(model_name,df,k_folds=10,tar_cols="", exclude_cols ="", saving_path="./", verbose=1):
    x = df.drop([tar_cols, exclude_cols], axis=1)
    y = df[tar_cols]
    # k_fold constructing the cross-validation framework
    skf = StratifiedKFold(k_fold = k_folds,shuffle=True, random_state=123 )
    model_name = model_name 
    for i, (train_index, test_index) in enumerate(skf.split(x,y)):   
        def objective(trial):
            clf = TabModelClassifier(n_d=trial.suggest_int("n_d", 8, 64),
                                    n_a =trial.suggest_int("n_a", 8, 64),
                                    n_steps = trial.suggest_int("n_steps",3,10),
                                    gamma =trial.suggest_float("gamma", 1.0, 2.0),
                                    n_independent = trial.suggest_int("n_independent",1,5),
                                    n_shared = trial.suggest_int("n_shared",1,5),
                                    momentum = trial.suggest_float("momentum", 0.01, 0.4),
                                    optimizer_fn = trial.suggest_categorical([torch.optim.Adam, torch.optim.Adadelta, torch.optim.Adagrad, torch.optim.Adamax, torch.optim.ASGD,torch.optim.SGD]),
                                    optimizer_params = dict(lr=trial.suggest_float("lr",1e-4,1e-3)),
                                    scheduler_fn = torch.optim.lr_scheduler,
                                    scheduler_params = {"gamma" :trial.suggest_float("sch-gamma", 0.5, 0.95), "step_size": trial.suggest_int("sch_step_size", 10, 20, 2)},
                                    model_name = model_name,
                                    saving_path = saving_path,
                                    verbose = verbose,
                                    device_name = "auto"
                                    )
            X_train,X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            clf.fit(X_train, Y_train,
                    eval_set = [(X_test, Y_test)],
                    eval_metrics = ['auc'])
            Y_pred = clf.predict(X_test)
            acc = accuracy_score(Y_pred, Y_test)
            return acc
    
        study = opt.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params

        clf_model = TabModelClassifier(best_params)
        make_save_cv_model(i,model_name,clf_model,best_params)


