from data import MultiModalDataset, collate_fn_batch_padding
from torch.utils.data import DataLoader
from tools import load_json, parse_gdl, parse_cdl, save_json
from problem import Problem
from tools import draw_dag, debug_execute
from tools import topological_sort, topological_sort_bidirectional
from copy import deepcopy


def show_theorem_len():
    len_to_pid = {}
    for pid in range(1, 7001):
        t_length = len(load_json(f'../../datasets/problems/{pid}.json')['theorem_seqs'])
        if t_length not in len_to_pid:
            len_to_pid[t_length] = [pid]
        else:
            len_to_pid[t_length].append(pid)
    count = 0
    for t_len in sorted(len_to_pid.keys()):
        count += len(len_to_pid[t_len])
        print(
            f'len={t_len}, count={len(len_to_pid[t_len])}, accu={count}({round(count / 70, 2)}%), pids={len_to_pid[t_len]}.')


def test_make_onehot():
    dataset = MultiModalDataset(problem_ids=list(range(1, 30)), name='t1')
    print('image', type(dataset[0]['image']), dataset[0]['image'].shape)
    print('state', type(dataset[0]['state']), dataset[0]['state'].shape)
    print('forward_theorem', type(dataset[0]['forward_theorem']), dataset[0]['forward_theorem'].shape)
    print('backward_theorem', type(dataset[0]['backward_theorem']), dataset[0]['backward_theorem'].shape)
    print()

    data_loader_train = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn_batch_padding,
        batch_size=16,
        shuffle=False
    )

    print('len(data_loader_train)', len(data_loader_train))
    print()

    for batch in data_loader_train:
        images = batch['images']
        states = batch['states']
        forward_theorems = batch['forward_theorems']
        backward_theorems = batch['backward_theorems']
        print('image', type(images), images.shape)
        print('state', type(states), states.shape)
        print('forward_theorem', type(forward_theorems), forward_theorems.shape)
        print('backward_theorem', type(backward_theorems), backward_theorems.shape)
        exit(0)


def run_bidirectional_solving():
    parsed_gdl = parse_gdl(load_json('../../datasets/gdl.json'))
    while True:
        pid = int(input('Input problem ID: '))
        # pid = 9
        cdl = load_json(f'../../datasets/problems/{pid}.json')
        parsed_cdl = parse_cdl(cdl)
        problem_initial = Problem(parsed_gdl, parsed_cdl)

        problem = deepcopy(problem_initial)
        for theorem in cdl['theorem_seqs']:
            problem.apply(theorem)
        # problem.show()

        direction = input("Input direction ('f', 'b' or 'h'): ")
        parameter_free = int(input("Parameterized (input 0) or parameter-free (input 1): ")) == 1
        # direction = 'b'
        # parameter_free = True

        forward_dag = cdl['forward_dag']
        for i in range(len(forward_dag['edges'])):
            forward_dag['edges'][i] = tuple(forward_dag['edges'][i])
        backward_dag = cdl['backward_dag']
        for i in range(len(backward_dag['edges'])):
            backward_dag['edges'][i] = tuple(backward_dag['edges'][i])
        save_json(forward_dag, '../../outputs/files/forward_dag.json')
        draw_dag(forward_dag, f'../../outputs/files/forward_dag')
        save_json(backward_dag, '../../outputs/files/backward_dag.json')
        draw_dag(backward_dag, f'../../outputs/files/backward_dag')

        if direction == 'f':
            while True:
                random_seed = int(input('Input random seed (-1 to exit): '))
                if random_seed == -1:
                    break
                problem = deepcopy(problem_initial)
                for theorem in topological_sort(forward_dag, random_seed):
                    if parameter_free:
                        theorem = theorem.split('(')[0]
                    debug_execute(problem.apply, [theorem])
                    # input('continue?:')
                    # problem.show()
                    # input('continue?:')

        elif direction == 'b':
            while True:
                random_seed = int(input("'Input random seed (-1 to exit): "))
                if random_seed == -1:
                    break

                problem = deepcopy(problem_initial)
                for theorem in topological_sort(backward_dag, random_seed):
                    if parameter_free:
                        theorem = theorem.split('(')[0]
                    debug_execute(problem.decompose, [theorem])
                    # input('continue?:')
                    # problem.show()
                    # input('continue?:')

        elif direction == 'h':
            while True:
                random_seed = int(input("'Input random seed (-1 to exit): "))
                if random_seed == -1:
                    break

                problem = deepcopy(problem_initial)
                hybrid_theorem_seqs = topological_sort_bidirectional(forward_dag, backward_dag, random_seed)
                save_json(hybrid_theorem_seqs, f'../../outputs/files/hybrid_theorem_seqs.json')
                for d, theorem in hybrid_theorem_seqs:
                    if parameter_free:
                        theorem = theorem.split('(')[0]
                    if d == 'f':
                        debug_execute(problem.apply, [theorem])
                    else:
                        debug_execute(problem.decompose, [theorem])
                    # input('continue?:')
                    # problem.show()
                    # input('continue?:')
        else:
            raise Exception(f'Unknown direction {direction}.')


if __name__ == '__main__':
    run_bidirectional_solving()
