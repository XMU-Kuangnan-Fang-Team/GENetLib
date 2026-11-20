import copy
from itertools import product
import math
import warnings

import pandas as pd
import torch

from GENetLib.pre_data import pre_data2
from GENetLib.scalar_l2train import scalar_l2train
from GENetLib.scalar_mcp_l2train import scalar_mcp_l2train
from GENetLib.survival_costfunc_cindex import neg_par_log_likelihood


"""Grid search for scalar_ge"""

pd.set_option("mode.chained_assignment", None)


def grid_scalar_ge(
    y,
    G,
    E,
    ytype,
    num_hidden_layers,
    nodes_hidden_layer,
    num_epochs,
    learning_rate1,
    learning_rate2,
    lambda1=None,
    lambda2=None,
    Lambda=None,
    threshold=None,
    model=None,
    split_type=0,
    ratio=[7, 3],
    important_feature=True,
    plot=True,
    model_reg=None,
    isfunc=False,
):
    # Grid candidates are complete two-stage training procedures. Pretraining
    # supplies a starting point; its validation loss does not eliminate a pair
    # of hidden-layer learning rate and ridge parameters before fine-tuning.
    def nonempty_grid(values, name):
        try:
            candidates = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be a nonempty iterable") from exc
        if not candidates:
            raise ValueError(f"{name} must be a nonempty iterable")
        return candidates

    learning_rate1 = nonempty_grid(learning_rate1, "learning_rate1")
    learning_rate2 = nonempty_grid(learning_rate2, "learning_rate2")
    lambda2 = nonempty_grid(lambda2, "lambda2")
    Lambda = nonempty_grid(Lambda, "Lambda")
    metric_names = {
        "Continuous": "mse",
        "Binary": "bce",
        "Survival": "neg_partial_log_likelihood",
    }
    if ytype not in metric_names:
        raise ValueError("Invalid ytype")
    entry_rng_state = torch.get_rng_state().clone()
    # Define dimensions for gene, environment and gene-environment features.
    In_Nodes = G.shape[1]
    Clinical_Nodes = E.shape[1]
    Interaction_Nodes = G.shape[1] * E.shape[1]
    x = G
    clinical = E
    interaction = None
    # Split data into training and validation(test) sets
    if split_type == 1:
        (
            x_train,
            y_train,
            clinical_train,
            interaction_train,
            x_valid,
            y_valid,
            clinical_valid,
            interaction_valid,
            x_test,
            y_test,
            clinical_test,
            interaction_test,
        ) = pre_data2(y, x, clinical, interaction, ytype, split_type, ratio)
    elif split_type == 0:
        (
            x_train,
            y_train,
            clinical_train,
            interaction_train,
            x_valid,
            y_valid,
            clinical_valid,
            interaction_valid,
        ) = pre_data2(y, x, clinical, interaction, ytype, split_type, ratio)
    else:
        raise ValueError("split_type must be 0 or 1")

    def validation_score(candidate):
        # Hyperparameters are compared by prediction error, without adding a
        # candidate-specific training penalty. Binary's legacy returned metric
        # remains accuracy, while its search criterion is minimized BCE.
        candidate.eval()
        with torch.no_grad():
            prediction = candidate(x_valid, interaction_valid, clinical_valid)
            if ytype == "Continuous":
                score = torch.nn.functional.mse_loss(prediction, y_valid)
            elif ytype == "Binary":
                score = torch.nn.functional.binary_cross_entropy(
                    prediction, y_valid
                )
            else:
                score = neg_par_log_likelihood(
                    prediction, y_valid[0], y_valid[1]
                )
        return float(score.detach().cpu().item())

    def parameters(lr2, ridge, mcp, lr_sparse):
        return {
            "learning_rate1": lr_sparse,
            "learning_rate2": lr2,
            "lambda1": lambda1,
            "lambda2": mcp,
            "Lambda": ridge,
        }

    search_results = []
    best_score = float("inf")
    GENet = None
    # Enumerate the full Cartesian product, reusing each (lr2, ridge)
    # pretraining fit for all independent (MCP, sparse-learning-rate) trials.
    for lr2, l2 in product(learning_rate2, Lambda):
        torch.set_rng_state(entry_rng_state)
        try:
            _, _, _, _, pretrained_model = scalar_l2train(
                x_train,
                clinical_train,
                interaction_train,
                y_train,
                x_valid,
                clinical_valid,
                interaction_valid,
                y_valid,
                In_Nodes,
                Interaction_Nodes,
                Clinical_Nodes,
                num_hidden_layers,
                nodes_hidden_layer,
                ytype,
                isfunc,
                lr2,
                l2,
                num_epochs,
                model_reg,
            )
        except Exception as exc:
            message = f"Pretraining failed for lr2={lr2}, Lambda={l2}: {exc}"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            for l, lrMCP in product(lambda2, learning_rate1):
                search_results.append({
                    "params": parameters(lr2, l2, l, lrMCP),
                    "score": None,
                    "status": "failed",
                    "error": message,
                })
            continue

        fine_tuning_rng_state = torch.get_rng_state().clone()
        for l, lrMCP in product(lambda2, learning_rate1):
            trial_parameters = parameters(lr2, l2, l, lrMCP)
            torch.set_rng_state(fine_tuning_rng_state)
            try:
                loss_train, loss_valid, index_tr, index_va, MCPNet = scalar_mcp_l2train(
                    x_train,
                    clinical_train,
                    interaction_train,
                    y_train,
                    x_valid,
                    clinical_valid,
                    interaction_valid,
                    y_valid,
                    In_Nodes,
                    Interaction_Nodes,
                    Clinical_Nodes,
                    num_hidden_layers,
                    nodes_hidden_layer,
                    ytype,
                    isfunc,
                    lr2,
                    l2,
                    lrMCP,
                    l,
                    num_epochs,
                    lambda1,
                    plot,
                    copy.deepcopy(pretrained_model),
                    model_reg,
                )
                score = validation_score(MCPNet)
                if not math.isfinite(score):
                    raise ValueError(f"Non-finite validation score: {score}")
            except Exception as exc:
                message = f"Fine-tuning failed for {trial_parameters}: {exc}"
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                search_results.append({
                    "params": trial_parameters,
                    "score": None,
                    "status": "failed",
                    "error": message,
                })
                continue

            search_results.append({
                "params": trial_parameters,
                "score": score,
                "status": "ok",
            })
            if score < best_score:
                best_score = score
                best_parameters = trial_parameters.copy()
                opt_L2 = l2
                opt_Learning_Rate2 = lr2
                opt_loss = loss_valid
                GENet = MCPNet
                opt_l = l
                opt_lr = lrMCP
                opt_loss_train = loss_train
                opt_index_tr = index_tr
                opt_index_va = index_va

    if GENet is None:
        error = ValueError(
            "No parameter combination produced a finite validation score; "
            "inspect the search_results_ attribute on this exception."
        )
        error.search_results_ = search_results
        raise error

    GENet.search_results_ = search_results
    GENet.best_params_ = best_parameters
    GENet.selection_metric_ = metric_names[ytype]
    GENet.selection_score_ = best_score

    # Define a function to identify important features based on a threshold t
    def important_features(tensor_, t):
        maxNum = max(abs(tensor_))
        resultPos = torch.where(abs(tensor_) > maxNum * t)[0].tolist()
        return resultPos

    if threshold != None:
        tensor1 = GENet.sparse1.weight.data
        tensor2 = GENet.sparse2.weight.data
        ifs_G = important_features(tensor1, threshold)
        ifs_GE = important_features(tensor2, threshold)
    # Print performance metrics
    if ytype == "Binary":
        print(
            "opt_index: L2:",
            opt_L2,
            "LR2:",
            opt_Learning_Rate2,
            "LR_MCP:",
            opt_lr,
            "L:",
            opt_l,
        )
        print(
            "Accuracy of train:", opt_loss_train, "Accuracy of valid:", opt_loss
        )
        print("AUC of train:", opt_index_tr, "AUC of valid:", opt_index_va)
        if threshold != None and important_feature == True:
            print("Important feature of gene:", ifs_G)
            print("Important feature of GE:", ifs_GE)
    elif ytype == "Continuous":
        print(
            "opt_index: L2:",
            opt_L2,
            "LR2:",
            opt_Learning_Rate2,
            "LR_MCP:",
            opt_lr,
            "L:",
            opt_l,
        )
        print(
            "MSE of train:",
            opt_loss_train.detach().numpy()[0],
            "MSE of valid:",
            opt_loss.detach().numpy()[0],
        )
        print("R2 of train:", opt_index_tr, "R2 of valid:", opt_index_va)
        if threshold != None and important_feature == True:
            print("Important feature of gene:", ifs_G)
            print("Important feature of GE:", ifs_GE)
    elif ytype == "Survival":
        print(
            "opt_index: L2:",
            opt_L2,
            "LR2:",
            opt_Learning_Rate2,
            "LR_MCP:",
            opt_lr,
            "L:",
            opt_l,
        )
        print(
            "Loss of train:",
            opt_loss_train.detach().numpy(),
            "Loss of valid:",
            opt_loss.detach().numpy(),
        )
        print(
            "C_index of train:",
            opt_index_tr.detach().numpy(),
            "C_index of valid:",
            opt_index_va.detach().numpy(),
        )
        if threshold != None and important_feature == True:
            print("Important feature of gene:", ifs_G)
            print("Important feature of GE:", ifs_GE)
    if threshold != None:
        return (
            [opt_L2, opt_Learning_Rate2, opt_lr, opt_l],
            opt_loss_train,
            opt_loss,
            opt_index_tr,
            opt_index_va,
            GENet,
            ifs_G,
            ifs_GE,
        )
    else:
        return (
            [opt_L2, opt_Learning_Rate2, opt_lr, opt_l],
            opt_loss_train,
            opt_loss,
            opt_index_tr,
            opt_index_va,
            GENet,
        )
