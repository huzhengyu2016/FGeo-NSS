from tools import save_pickle, parse_gdl, parse_cdl, delete_node_from_dag, load_pickle
from tools import state_letters, theorem_letters
from problem import Problem
from PIL import Image
import numpy as np
from copy import deepcopy
import time
from tools import config, load_json, save_json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import warnings
from multiprocessing import Process, Queue
import random
import psutil
import argparse


def make_onehot_image(pid):
    image = Image.open(f'../../datasets/diagrams/{pid}.png').convert('L')
    image = image.resize((config['data']['image_size'], config['data']['image_size']))
    image = np.array(image).astype(np.float32) / 255.0  # (256, 256)

    h_patches = config['data']['image_size'] // config['data']['patch_size']  # 16
    w_patches = h_patches  # 16

    # reshape to [h_patches, patch_size, w_patches, patch_size] = (16, 16, 16, 16)
    patches = image.reshape(h_patches, config['data']['patch_size'], w_patches, config['data']['patch_size'])

    # [h_patches, patch_size, w_patches, patch_size] -> [h_patches, w_patches, patch_size, patch_size]
    patches = patches.transpose(0, 2, 1, 3)  # (16, 16, 16, 16)

    # flatten
    patches = patches.reshape(-1, config['data']['patch_size'], config['data']['patch_size'])  # (256, 16, 16)
    patches = patches.reshape(patches.shape[0], -1)  # (256, 16)
    patches = torch.from_numpy(patches).float()
    # print(type(patches), patches.shape)

    return patches


def make_onehot_state(state):
    if len(state) > config['data']['max_state_length']:
        i = int(config['data']['max_state_length'] * 0.3) - 1
        j = len(state) - int(config['data']['max_state_length'] * 0.3) - 1
        selected = sorted(random.sample(
            list(range(i + 1, j)), config['data']['max_state_length'] - len(state[:i + 1]) - len(state[j:])))
        selected_state = [state[s] for s in selected]
        state = state[:i + 1] + selected_state + state[j:]
    return torch.tensor([state_letters.index(token) for token in state], dtype=torch.int)


class MultiModalDataset(Dataset):
    def __init__(self, problem_ids, name):
        self.problem_ids = problem_ids
        self.name = name
        self.images = []
        self.state_id_to_image_id = []
        self.states = []
        self.forward_theorems = []
        self.backward_theorems = []
        self.length_of_state = {}  # {length: count}

        print(f'loading {name} sets...')
        if os.path.exists(f'../../outputs/synthetic_data/{self.name}.pk'):
            data = load_pickle(f'../../outputs/synthetic_data/{self.name}.pk')
            (self.images, self.state_id_to_image_id, self.states,
             self.forward_theorems, self.backward_theorems, self.length_of_state) = data
            return

        count = 0
        for filename in os.listdir('../../outputs/synthetic_data'):
            if '-' not in filename:
                continue
            pid = int(filename.split('-')[0])
            if pid not in self.problem_ids:
                continue

            try:
                image_id = len(self.images)
                self.images.append(make_onehot_image(pid))

                for state, forward_theorem, backward_theorem in load_pickle(f'../../outputs/synthetic_data/{filename}'):
                    state_onehot = make_onehot_state(state)
                    forward_theorem_onehot = torch.zeros(len(theorem_letters))
                    for theorem in forward_theorem:
                        forward_theorem_onehot[theorem_letters.index(theorem)] = 1
                    backward_theorem_onehot = torch.zeros(len(theorem_letters))
                    for theorem in backward_theorem:
                        backward_theorem_onehot[theorem_letters.index(theorem)] = 1

                    self.state_id_to_image_id.append(image_id)
                    self.states.append(state_onehot)
                    self.forward_theorems.append(forward_theorem_onehot)
                    self.backward_theorems.append(backward_theorem_onehot)

                    if len(state_onehot) not in self.length_of_state:
                        self.length_of_state[len(state_onehot)] = 1
                    else:
                        self.length_of_state[len(state_onehot)] += 1
            except BaseException as e:
                print(f'problem {pid} raise Exception:', repr(e))

            count += 1
            print(f"Loading problem {pid} in set {self.name} ({count} / {len(self.problem_ids)})")

        data = (self.images, self.state_id_to_image_id, self.states,
                self.forward_theorems, self.backward_theorems, self.length_of_state)
        save_pickle(data, f'../../outputs/synthetic_data/{self.name}.pk')

    def __len__(self):
        return len(self.states)

    def show(self):
        print('name:', self.name)
        print('number:', len(self.states))
        total_length_count = 0
        for length in self.length_of_state:
            total_length_count += self.length_of_state[length]
        accumulate_length_count = 0
        for length, length_count in sorted(self.length_of_state.items(), key=lambda x: x[0]):
            accumulate_length_count += length_count
            print(length, length_count, round(accumulate_length_count / total_length_count, 4))

    def __getitem__(self, idx):
        return {
            'image': self.images[self.state_id_to_image_id[idx]],
            'state': self.states[idx],
            'forward_theorem': self.forward_theorems[idx],
            'backward_theorem': self.backward_theorems[idx]
        }


def collate_fn_batch_padding(batch):
    images = []
    states = []
    forward_theorems = []
    backward_theorems = []

    for item in batch:
        images.append(item['image'])
        states.append(item['state'])
        forward_theorems.append(item['forward_theorem'])
        backward_theorems.append(item['backward_theorem'])

    images = torch.stack(images)
    states = pad_sequence(
        states,
        batch_first=True,
        padding_value=0
    )
    forward_theorems = torch.stack(forward_theorems)
    backward_theorems = torch.stack(backward_theorems)

    return {
        'images': images,
        'states': states,
        'forward_theorems': forward_theorems,
        'backward_theorems': backward_theorems
    }


def make_train_val_test_split():
    filename = "../../outputs/log/log_data_problem_split.json"
    if os.path.exists(filename):
        return load_json(filename)

    problem_ids = list(range(1, config['data']['max_pid'] + 1))
    random.Random(config['data']['random_seed']).shuffle(problem_ids)

    train, val, test = [int(n) for n in config['data']['train_val_test_split'].split(':')]
    train_problem_ids = sorted(problem_ids[:int(config['data']['max_pid'] * train / (train + val + test))])
    val_problem_ids = sorted(problem_ids[int(config['data']['max_pid'] * train / (train + val + test)):
                                         int(config['data']['max_pid'] * (train + val) / (train + val + test))])
    test_problem_ids = sorted(problem_ids[int(config['data']['max_pid'] * (train + val) / (train + val + test)):])
    problem_split = {"train": train_problem_ids, "val": val_problem_ids, "test": test_problem_ids}

    print(f"train: {len(train_problem_ids)}, val: {len(val_problem_ids)}, test: {len(test_problem_ids)}")
    save_json(problem_split, filename)

    return problem_split


def multiprocess_generate(task_queue, reply_queue, parameter_free):
    warnings.filterwarnings("ignore")
    parsed_gdl = parse_gdl(load_json('../../datasets/gdl.json'))
    while not task_queue.empty():
        problem_id = task_queue.get()
        timing = time.time()
        try:
            synthetic_data = []  # (state, forward_theorems, backward_theorems)
            rng = random.Random(config['data']['random_seed'])
            cdl = load_json(f'../../datasets/problems/{problem_id}.json')
            for i in range(len(cdl['forward_dag']['edges'])):
                cdl['forward_dag']['edges'][i] = tuple(cdl['forward_dag']['edges'][i])
            for i in range(len(cdl['backward_dag']['edges'])):
                cdl['backward_dag']['edges'][i] = tuple(cdl['backward_dag']['edges'][i])
            parsed_cdl = parse_cdl(cdl)
            problem_initial = Problem(parsed_gdl, parsed_cdl)

            repeat_count = 0
            correct_count = 0
            repeat_total = len(cdl['theorem_seqs']) * config['data']['repeat_rate']
            while repeat_count < repeat_total:
                state_and_action = []
                forward_dag = deepcopy(cdl['forward_dag'])
                backward_dag = deepcopy(cdl['backward_dag'])
                problem = deepcopy(problem_initial)
                directions = []
                theorem_seqs = []
                forward_candidates = [node for node in forward_dag['in_degree']
                                      if forward_dag['in_degree'][node] == 0 and node not in theorem_seqs]
                backward_candidates = [node for node in backward_dag['in_degree']
                                       if backward_dag['in_degree'][node] == 0 and node not in theorem_seqs]
                while len(forward_candidates) > 0 or len(backward_candidates) > 0:
                    state_and_action.append((
                        problem.state(),
                        [theorem.split('(')[0] for theorem in forward_candidates],
                        [theorem.split('(')[0] for theorem in backward_candidates]
                    ))

                    if rng.random() < 0.5:
                        theorem = rng.choice(forward_candidates)
                        if parameter_free:
                            theorem = theorem.split('(')[0]
                        problem.apply(theorem)
                        directions.append('f')
                        theorem_seqs.append(theorem)
                    else:
                        theorem = rng.choice(backward_candidates)
                        if parameter_free:
                            theorem = theorem.split('(')[0]
                        problem.decompose(theorem)
                        directions.append('b')
                        theorem_seqs.append(theorem)

                    delete_node_from_dag(theorem, forward_dag)
                    delete_node_from_dag(theorem, backward_dag)

                    forward_candidates = [node for node in forward_dag['in_degree']
                                          if forward_dag['in_degree'][node] == 0 and node not in theorem_seqs]
                    backward_candidates = [node for node in backward_dag['in_degree']
                                           if backward_dag['in_degree'][node] == 0 and node not in theorem_seqs]

                repeat_count += 1
                if problem.status_of_goal[0] == 1:
                    synthetic_data.extend(state_and_action)
                    correct_count += 1

        except BaseException as e:
            reply_queue.put((os.getpid(), problem_id, 'error', repr(e), time.time() - timing))
        else:
            if len(synthetic_data) > 0:
                save_pickle(synthetic_data, (f"../../outputs/synthetic_data/{problem_id}-rp_{repeat_total}-"
                                             f"rs_{config['data']['random_seed']}.pk"))
                reply_queue.put((os.getpid(), problem_id, 'generated',
                                 f"repeat={correct_count}/{repeat_total}, state-action_number={len(synthetic_data)}",
                                 time.time() - timing))
            else:
                reply_queue.put((os.getpid(), problem_id, 'error',
                                 "No correct state-action.", time.time() - timing))


def generate_synthetic_data(solve_again, parameter_free):
    log_path = '../../outputs/log/log_data_generate_synthetic_data.json'
    log_synthetic_data = {'generated': {}, 'error': {}}
    if os.path.exists(log_path):
        log_synthetic_data = load_json(log_path)

    problem_ids = []
    for pid in range(config['data']['max_pid']):
        pid += 1
        if str(pid) in log_synthetic_data['generated']:
            continue
        if not solve_again and str(pid) in log_synthetic_data['error']:
            continue
        problem_ids.append(pid)

    task_queue = Queue()
    random.shuffle(problem_ids)
    for problem_id in problem_ids:
        task_queue.put(problem_id)

    reply_queue = Queue()
    process_ids = []
    max_process_count = int(config['data']['multiprocess_rate'] * psutil.cpu_count())

    count = 0
    while True:
        for process_id in process_ids.copy():  # remove non-existent pid
            alive = True
            try:  # check whether process is alive
                process = psutil.Process(process_id)
                if process.status() in [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD]:
                    alive = False
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                process_ids.remove(process_id)

        while not task_queue.empty() and max_process_count - len(process_ids) > 0:
            process = Process(
                target=multiprocess_generate,
                args=(task_queue, reply_queue, parameter_free)
            )
            process.start()
            process_ids.append(process.pid)

        if not reply_queue.empty():  # directly calling .get() will block process
            process_id, problem_id, result, msg, timing = reply_queue.get()
            log_synthetic_data[result][problem_id] = msg
            count += 1
            print(f"({count}/{len(problem_ids)}) process_id={process_id}, problem_id={problem_id}, "
                  f"result={result}, msg={msg}, timing={timing}")
            save_json(log_synthetic_data, log_path)


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")
    parser.add_argument("--func", type=str, required=True,
                        choices=["generate_synthetic_data", "make_training_data", ],
                        help="function that you want to run")
    parser.add_argument("--solve_again", action="store_true", default=False)
    parser.add_argument("--parameter_free", action="store_true", default=False)

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


def make_training_data():
    problem_split = make_train_val_test_split()  # data
    train = MultiModalDataset(problem_split['train'], 'train')
    val = MultiModalDataset(problem_split['val'], 'val')
    train.show()
    print()
    val.show()
    print()
    input('continue?:')


if __name__ == '__main__':
    """
    python tools.py --func kill --filename data.py
    
    python data.py --func generate_synthetic_data
    python data.py --func make_training_data
    """
    args = get_args()
    if args.func == "generate_synthetic_data":
        generate_synthetic_data(args.solve_again, args.parameter_free)
    elif args.func == "make_training_data":
        make_training_data()
