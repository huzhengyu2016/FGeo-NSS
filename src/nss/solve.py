from tools import load_json, config, save_json, parse_gdl, parse_cdl, parse_fact
from problem import Problem
from multiprocessing import Process, Queue
from func_timeout import func_timeout, FunctionTimedOut
import torch
import warnings
import random
import time
import argparse
import os
from data import make_train_val_test_split, make_onehot_image, make_onehot_state
from tools import get_theorem_seqs, theorem_letters
from model import make_model
import psutil
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
import traceback


def solve(problem_id, parsed_gdl, request_queue, reply_queue, forward_only, beam_size):
    cdl = load_json(f'../../datasets/problems/{problem_id}.json')
    theorems_gt = {}
    for t in cdl['theorem_seqs']:
        name, paras = parse_fact(t)
        if name in theorems_gt:
            theorems_gt[name].append(paras)
        else:
            theorems_gt[name] = [paras]
    parsed_cdl = parse_cdl(cdl)

    beam_stacks = [(Problem(parsed_gdl, parsed_cdl), 1)]  # (problem, prob)
    while len(beam_stacks) > 0:
        states = [problem.state() for problem, _ in beam_stacks]
        request_queue.put((os.getpid(), problem_id, "predict", states))
        reply_problem_id, theorems_pd = None, None
        while reply_problem_id is None or reply_problem_id != problem_id:
            if not reply_queue.empty():
                reply_problem_id, theorems_pd = reply_queue.get()

        if forward_only:  # apply theorem
            theorems_pd = theorems_pd.softmax(dim=1)  # convert to a probability distribution
            for i in range(len(beam_stacks)):  # consider previous prob
                theorems_pd[i] = theorems_pd[i] * beam_stacks[i][1]
        else:
            theorems_pd_forward, theorems_pd_backward = theorems_pd
            theorems_pd_forward = theorems_pd_forward.softmax(dim=1)
            theorems_pd_backward = theorems_pd_backward.softmax(dim=1)
            for i in range(len(beam_stacks)):  # consider previous prob
                theorems_pd_forward[i] = theorems_pd_forward[i] * beam_stacks[i][1]
                theorems_pd_backward[i] = theorems_pd_backward[i] * beam_stacks[i][1]
            theorems_pd = torch.cat([theorems_pd_forward, theorems_pd_backward], dim=0)

        theorems_pd = theorems_pd.flatten()  # flatten for sorting
        sorted_theorems = sorted(enumerate(theorems_pd), key=lambda x: x[1], reverse=True)  # (idx, prob)

        new_beam_stacks = []
        prob_sum = 0
        for idx, prob in sorted_theorems:
            if len(new_beam_stacks) == beam_size:  # max_len(beam_stack) = beam_size
                break
            beam_id = int(idx / len(theorem_letters))
            theorem_id = idx % len(theorem_letters)
            forward = True
            if beam_id >= len(beam_stacks):
                beam_id = beam_id % len(beam_stacks)
                forward = False

            problem = deepcopy(beam_stacks[beam_id][0])

            theorem = theorem_letters[theorem_id]

            # if theorem not in theorems_gt:
            #     continue

            if theorem in theorems_gt:  # add para
                theorems = [theorem + '(' + ','.join(para) + ')' for para in theorems_gt[theorem]]
            else:
                theorems = [theorem]

            update = False
            if forward:
                for theorem in theorems:
                    update = problem.apply(theorem) or update
            else:
                for theorem in theorems:
                    update = problem.decompose(theorem) or update

            if update:
                if problem.status_of_goal[0] == 1:
                    seqs = get_theorem_seqs(problem)
                    return "solved", seqs
                new_beam_stacks.append([problem, prob])
                prob_sum += prob

        for i in range(len(new_beam_stacks)):
            problem, prob = new_beam_stacks[i]
            prob = prob / prob_sum  # probability normalization
            new_beam_stacks[i] = (problem, prob)

        beam_stacks = new_beam_stacks

    return "unsolved", "Out of stacks."


def multiprocess_solve(task_queue, request_queue, reply_queue, forward_only, beam_size, timeout):
    warnings.filterwarnings("ignore")
    parsed_gdl = parse_gdl(load_json('../../datasets/gdl.json'))
    while not task_queue.empty():
        problem_id = task_queue.get()

        timing = time.time()
        try:
            solved, msg = func_timeout(
                timeout=timeout,
                func=solve,
                args=(problem_id, parsed_gdl, request_queue, reply_queue, forward_only, beam_size)
            )
            info = (solved, msg, time.time() - timing)
        except FunctionTimedOut:
            info = ("timeout", f"FunctionTimedOut({timeout})", time.time() - timing)
        except BaseException as e:
            info = ("error", traceback.format_exc(), time.time() - timing)
            # info = ("error", e, time.time() - timing)

        request_queue.put((os.getpid(), problem_id, "write", info))


def main(device, test_pids, solve_again, timeout, text_only, forward_only, no_gate, small_model, beam_size):
    flag = str(text_only)[0] + str(forward_only)[0] + str(no_gate)[0] + str(small_model)[0]
    model_bst_path = f"../../outputs/checkpoints/{flag}_model_bst.pth"
    flag += f"_bs{beam_size}"
    flag += f"_tm{timeout}"
    log_path = f"../../outputs/log/log_pssr_nss_{flag}.json"

    log = {"total": test_pids, "solved": {}, "unsolved": {}, "timeout": {}, "error": {}}
    if os.path.exists(log_path):
        log = load_json(log_path)

    task_queue = Queue()
    problem_ids = []
    generated_pids = load_json('../../outputs/log/log_data_generate_synthetic_data.json')['generated']
    if solve_again:  # clear unsolved, timeout, and error, run again
        log["unsolved"] = {}
        log["timeout"] = {}
        log["error"] = {}

    for problem_id in test_pids:
        if str(problem_id) not in generated_pids:  # skip problem with wrong annotations
            continue
        if str(problem_id) in log["solved"]:
            continue
        if str(problem_id) in log["unsolved"]:
            continue
        if str(problem_id) in log["timeout"]:
            continue
        if str(problem_id) in log["error"]:
            continue
        problem_ids.append(problem_id)

    random.shuffle(problem_ids)
    for problem_id in problem_ids:
        task_queue.put(problem_id)
    request_queue = Queue()
    reply_queues = {}  # map pid to process queue

    images_cache = {}  # pid to image_one_hot

    model = make_model(text_only, forward_only, no_gate, small_model)
    model.load_state_dict(torch.load(model_bst_path, map_location=torch.device(device), weights_only=True))
    model.eval()
    model = model.to(device)
    count = 0
    max_process_count = int(config['test']['multiprocess_rate'] * psutil.cpu_count())
    while True:
        for process_id in list(reply_queues.keys()):  # remove non-existent pid
            alive = True
            try:  # check whether process is alive
                process = psutil.Process(process_id)
                if process.status() in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                    alive = False
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                del reply_queues[process_id]

        while not task_queue.empty() and max_process_count - len(reply_queues) > 0:
            reply_queue = Queue()
            process = Process(
                target=multiprocess_solve,
                args=(task_queue, request_queue, reply_queue, forward_only, beam_size, timeout)
            )
            process.start()
            reply_queues[process.pid] = reply_queue

        if not request_queue.empty():  # directly calling .get() will block process
            process_id, problem_id, request, info = request_queue.get()
            if request == "write":  # write log
                count += 1
                result, msg, timing = info
                log[result][problem_id] = {"msg": msg, "timing": timing}
                if result == "error":
                    print('problem_id:', problem_id)
                    print(msg)
                save_json(log, log_path)
                print(f"({count}/{len(problem_ids)}) process_id={process_id}, problem_id={problem_id}, "
                      f"result={result}, timing={timing}")
            elif request == "predict":  # predict theorem
                if problem_id not in images_cache:
                    images_cache[problem_id] = make_onehot_image(problem_id)

                images = torch.stack([images_cache[problem_id] for _ in range(len(info))])
                states = pad_sequence(
                    [make_onehot_state(state) for state in info],
                    batch_first=True,
                    padding_value=0
                )

                reply_queue = reply_queues[process_id]
                with torch.no_grad():
                    if forward_only:
                        predictions_forward = model(states.to(device), images.to(device)).cpu()
                        reply_queue.put((problem_id, predictions_forward))
                    else:
                        predictions_forward, predictions_backward = model(states.to(device), images.to(device))
                        predictions_forward = predictions_forward.cpu()
                        predictions_backward = predictions_backward.cpu()
                        reply_queue.put((problem_id, (predictions_forward, predictions_backward)))

        if count == len(problem_ids):
            break


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text_only", action="store_true", default=False)
    parser.add_argument("--forward_only", action="store_true", default=False)
    parser.add_argument("--no_gate", action="store_true", default=False)
    parser.add_argument("--small_model", action="store_true", default=False)

    parser.add_argument("--solve_again", action="store_true", default=False)

    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == '__main__':
    """
    python tools.py --func kill --filename solve.py  # kill subprocess
    
    python solve.py --timeout 600 --solve_again --device cuda:1
    python solve.py  --device cuda:1
    python solve.py --text_only --device cuda:1
    python solve.py --forward_only --device cuda:1
    python solve.py --no_gate --device cuda:1
    python solve.py --small_model --device cuda:1
    """
    args = get_args()
    for bs in [5, 3, 1]:
        main(
            device=args.device, test_pids=make_train_val_test_split()['test'],
            solve_again=args.solve_again, timeout=args.timeout,
            text_only=args.text_only, forward_only=args.forward_only,
            no_gate=args.no_gate, small_model=args.small_model,
            beam_size=bs
        )
