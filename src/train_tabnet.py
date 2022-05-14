import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score,classification_report
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

def train(model_name,sc_df,tar_col,optim,k_folds=10,tar_cols="",verbose=1):

    ''' this function is used to train the model with parameters optimization using optuna and cross validation using stratified k_folds'''

    y = sc_df[tar_col]
    x = sc_df.drop([tar_col],axis=1)
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
                                    # scheduler_fn = torch.optim.lr_scheduler,
                                    # scheduler_params = {"gamma" :trial.suggest_float("sch-gamma", 0.5, 0.95), "step_size": trial.suggest_int("sch_step_size", 10, 20, 2)},
                                    verbose = verbose,
                                    device_name = "auto"
                                    )
            # print(f" train_index :: {train_index}")
            # print(f" test_index :: {test_index}")
            X_train,X_test = x.iloc[train_index,:], x.iloc[test_index,:]
            # print(X_train.shape, X_test.shape)
            X_train, X_test = X_train.to_numpy(dtype=np.float64), X_test.to_numpy(dtype=np.float64)
            Y_train, Y_test = y.iloc[train_index], y.iloc[test_index]
            Y_train, Y_test = Y_train.to_numpy(dtype=np.float64), Y_test.to_numpy(dtype=np.float64)
            print(Y_train.shape, Y_test.shape)
            clf.fit(X_train, Y_train,
                    eval_set=[(X_test, Y_test)],
                    eval_metric=['accuracy'])
            Y_pred = clf.predict(X_test)
            print(classification_report(Y_test, Y_pred, labels=[x for x in range(6)]))
            clf_report = classification_report(Y_test, Y_pred, labels=[x for x in range(6)])
            joblib.dump(clf_report,f"../outputs/classification_report/comp/{i}_{model_name}_classification_report.z")
            with open(f"../outputs/classification_report/{model_name}_{i}_classification_report.txt","w+") as file:file.write(clf_report)
            print(f"Saved classification_report at : outputs/classification_report/{model_name}_{i}_classification_report.txt")
            acc = accuracy_score(Y_pred, Y_test)
            return acc

        print(f"Starting optimization for fold : [{i}/{k_folds}]")
        study = opt.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        best_params = study.best_params
        print(f" Best params for fold : [{i}/{k_folds}]")
        print(best_params)
        joblib.dump(best_params,f"../outputs/{model_name}/best_params/comp/fold_{i}_best_params.z")
        with open(f"../outputs/{model_name}/best_params/fold_{i}_best_params.txt", "w+") as file:file.write(best_params)
        print(f"Saved best_params at : outputs/{model_name}/best_params/fold_{i}_best_params.txt")
        clf_model = TabNetClassifier(best_params)
        try:
            print("[++] Saving the model and parameters in corresponding directories")
            make_save_cv_model(i,model_name,clf_model,best_params,optim=optim)
        except:
            print("[-] Failed to save the model")





if __name__ == '__main__':
    use_df = pd.read_csv("../outputs/data/trainable_scaled.csv")
    tar_col = "PCE_categorical"
    model_name = "pytorch_tabnet"
    optimizer = "Adam"
    folds = 7
    # clf = TabNetClassifier()
    # y = use_df[tar_col]
    # x = use_df.drop([tar_col],axis=1)
    # X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2)
    # clf.fit(X_train,Y_train)
    # y_pred = clf.predict(X_test)
    # accuracy_score = accuracy_score(y_pred, Y_test)
    # print(accuracy_score) 
    # print("[++] Starting the training process ...")
    train(model_name=model_name,
        sc_df=use_df,
        tar_col=tar_col,
        optim=optimizer,
        k_folds=folds)
    print("[++] Ended the training process ...")