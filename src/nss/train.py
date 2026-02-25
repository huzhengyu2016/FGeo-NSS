import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import argparse
from model import make_model
from data import make_train_val_test_split, MultiModalDataset, collate_fn_batch_padding
from tools import config
from tools import load_json, save_json
import random

torch.manual_seed(config['training']['random_seed'])
random.seed(config['training']['random_seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config['training']['random_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_one_epoch(model, data_loader, loss_func, optimizer, flag, mode, epoch, epochs, forward_only, log):
    time.sleep(1)
    timing = time.time()

    if optimizer is not None:
        model.train()
    else:
        model.eval()
    device = next(model.parameters()).device

    loss_list = []  # for acv_loss calculation
    acc_count = {
        'forward_total': 0,
        'forward_correct': 0,
        'backward_total': 0,
        'backward_correct': 0
    }
    forward_results = []  # decoding results
    backward_results = []  # decoding results

    def calculation(ground_truth, prediction, forward=True):
        for i in range(len(ground_truth)):
            gt = [idx for idx, t in enumerate(ground_truth[i]) if t != 0]
            pd = [idx for idx, t in sorted(enumerate(prediction[i]), key=lambda x: x[1], reverse=True)][:len(gt)]
            if forward:
                acc_count['forward_total'] += len(gt)
                acc_count['forward_correct'] += len(set(gt) & set(pd))
                forward_results.append(f"direction=forward,\tGT[{', '.join([str(idx) for idx in gt])}],"
                                       f"\tPD=[{', '.join([str(idx) for idx in pd])}]")
            else:
                acc_count['backward_total'] += len(gt)
                acc_count['backward_correct'] += len(set(gt) & set(pd))
                backward_results.append(f"direction=backward,\tGT[{', '.join([str(idx) for idx in gt])}],"
                                        f"\tPD=[{', '.join([str(idx) for idx in pd])}]")

    loop = tqdm(data_loader, leave=False)  # running loop
    loop.set_description(f'{flag}-{mode} (epoch [{epoch}/{epochs}])')
    for batch in loop:
        images = batch['images'].to(device)
        states = batch['states'].to(device)
        forward_theorems_gt = batch['forward_theorems'].to(device)
        backward_theorems_gt = batch['backward_theorems'].to(device)

        if forward_only:
            forward_theorems_pd = model(state=states, image=images)  # predictions
            loss = loss_func(forward_theorems_pd.float(), forward_theorems_gt.float())  # loss
            loss_list.append(loss.item())
            if optimizer is not None:  # loss backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            calculation(forward_theorems_gt, forward_theorems_pd, forward=True)

        else:
            forward_theorems_pd, backward_theorems_pd = model(state=states, image=images)  # predictions

            loss = (loss_func(forward_theorems_pd.float(), forward_theorems_gt.float()) +  # loss
                    loss_func(backward_theorems_pd.float(), backward_theorems_gt.float()))
            loss_list.append(loss.item())
            if optimizer is not None:  # loss backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            calculation(forward_theorems_gt, forward_theorems_pd, forward=True)
            calculation(backward_theorems_gt, backward_theorems_pd, forward=False)

        loop.set_postfix(loss=loss.item())

    loop.close()

    avg_loss = round(sum(loss_list) / len(loss_list), 4)
    forward_acc = round(acc_count['forward_correct'] / acc_count['forward_total'], 4)
    with open(f"../../outputs/log/{flag}_forward_{mode}-{epoch}.txt", 'w', encoding='utf-8') as file:
        file.write("\n".join(forward_results))
        file.write(f"\navg_loss: {avg_loss}, forward_acc: {forward_acc}")
    backward_acc = 0
    timing = round(time.time() - timing, 4)

    if not forward_only:
        backward_acc = round(acc_count['backward_correct'] / acc_count['backward_total'], 4)
        with open(f"../../outputs/log/{flag}_backward_{mode}-{epoch}.txt", 'w', encoding='utf-8') as file:
            file.write("\n".join(backward_results))
            file.write(f"\navg_loss: {avg_loss}, backward_acc: {backward_acc}")

    log["log"][epoch][mode] = {
        "avg_loss": avg_loss, "forward_acc": forward_acc, "backward_acc": backward_acc, "timing": timing
    }

    print(f"{flag}-{mode}  (Epoch [{epoch}/{epochs}]): "
          f"avg_loss={avg_loss}, forward_acc={forward_acc}, backward_acc={backward_acc}, timing={timing}s.")

    return avg_loss


def train(device, text_only, forward_only, no_gate, small_model):
    flag = str(text_only)[0] + str(forward_only)[0] + str(no_gate)[0] + str(small_model)[0]
    log_path = f"../../outputs/log/{flag}_log_model_training.json"
    log = {
        "next_epoch": 1,
        "best_epoch": 0,
        "best_loss": 1e10,
        "log": {}
    }
    model_bst_path = f"../../outputs/checkpoints/{flag}_model_bst.pth"
    model_bk_path = f"../../outputs/checkpoints/{flag}_model_bk.pth"
    optimizer_bk_path = f"../../outputs/checkpoints/{flag}_optimizer_bk.pth"

    model = make_model(text_only, forward_only, no_gate, small_model).to(device)  # model

    problem_split = make_train_val_test_split()  # data
    data_loader_train = DataLoader(
        dataset=MultiModalDataset(problem_split['train'], 'train'),
        collate_fn=collate_fn_batch_padding,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    data_loader_val = DataLoader(
        dataset=MultiModalDataset(problem_split['val'], 'val'),
        collate_fn=collate_fn_batch_padding,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False
    )
    # data_loader_test = DataLoader(
    #     dataset=MultiModalDataset(problem_split['test'], 'test'),
    #     collate_fn=collate_fn_batch_padding,
    #     batch_size=config["training"]["batch_size"] * 2,
    #     shuffle=False
    # )

    loss_func = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss

    optimizer = torch.optim.Adam(  # Adam optimizer
        model.parameters(),
        lr=config["training"]["lr"]
    )

    if os.path.exists(log_path):
        log = load_json(log_path)
        model.load_state_dict(torch.load(model_bk_path, map_location=torch.device(device), weights_only=True))
        optimizer.load_state_dict(torch.load(optimizer_bk_path, map_location=torch.device(device), weights_only=True))

    epochs = config["training"]["epochs"]
    for epoch in range(log["next_epoch"], epochs + 1):
        # log['log'][epoch] = {'train': {}, 'val': {}, 'test': {}}
        log['log'][epoch] = {'train': {}, 'val': {}}

        run_one_epoch(  # train
            model=model, data_loader=data_loader_train, loss_func=loss_func, optimizer=optimizer,
            flag=flag, mode='train', epoch=epoch, epochs=epochs, forward_only=forward_only, log=log
        )

        with torch.no_grad():
            avg_loss = run_one_epoch(  # val
                model=model, data_loader=data_loader_val, loss_func=loss_func, optimizer=None,
                flag=flag, mode='val', epoch=epoch, epochs=epochs, forward_only=forward_only, log=log
            )
            if avg_loss < log["best_loss"]:
                log["best_epoch"] = epoch
                log["best_loss"] = avg_loss
                torch.save(model.state_dict(), model_bst_path)

            # run_one_epoch(  # test
            #     model=model, data_loader=data_loader_test, loss_func=loss_func, optimizer=None,
            #     flag=flag, mode='test', epoch=epoch, epochs=epochs, forward_only=forward_only, log=log
            # )

        torch.save(model.state_dict(), model_bk_path)
        torch.save(optimizer.state_dict(), optimizer_bk_path)
        log["next_epoch"] += 1
        save_json(log, log_path)


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use FGeo-NSS!")
    parser.add_argument("--device", type=str, required=False, default="cuda:0")
    parser.add_argument("--text_only", action="store_true", default=False)
    parser.add_argument("--forward_only", action="store_true", default=False)
    parser.add_argument("--no_gate", action="store_true", default=False)
    parser.add_argument("--small_model", action="store_true", default=False)

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == "__main__":
    """
    python train.py
    python train.py --text_only --device cuda:1
    python train.py --forward_only
    python train.py --no_gate --device cuda:1
    python train.py --small_model
    """
    args = get_args()
    train(args.device, args.text_only, args.forward_only, args.no_gate, args.small_model)
