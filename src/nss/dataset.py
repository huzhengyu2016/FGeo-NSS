from tools import load_json, parse_gdl, parse_cdl, save_json
from problem import Problem
from xml.etree import ElementTree
import warnings
import zipfile
import cv2
import os
import time
from tools import get_forward_dag, get_backward_dag, get_cleaned_theorem_seqs
from copy import deepcopy
from tools import config
from multiprocessing import Process, Queue
import random
import psutil
from func_timeout import func_timeout, FunctionTimedOut
import argparse


def find_points_from_ggb(ggb_filename):
    with zipfile.ZipFile(ggb_filename, 'r') as zip_ref:
        root = ElementTree.fromstring(zip_ref.read('geogebra.xml'))
        visible_points = {}
        for element in root.findall('.//element'):
            if element.get('type') == 'point':
                name = element.get('label')
                show = element.find('show')
                if show.get('object') == 'true' and show.get('label') == 'true':
                    caption = element.find('caption')
                    if caption is not None:  # point has other name
                        name = caption.get('val')
                    coords = element.find('coords')
                    x = coords.get('x')
                    y = coords.get('y')
                    z = coords.get('z')
                    if x != 'NaN' and y != 'NaN':
                        if z != "1":
                            visible_points[name] = (float(x) / float(z), float(y) / float(z))
                        else:
                            visible_points[name] = (float(x), float(y))

        return visible_points


def get_points_in_cons(construction_cdl):
    points_in_cons = set()
    for cons in construction_cdl:
        if cons.startswith('Shape'):
            cons = cons.replace('Shape(', '').replace(')', '').split(',')
            for item in cons:
                if len(item) == 1:
                    points_in_cons.add(item)
                elif len(item) == 2:
                    points_in_cons.add(item[0])
                    points_in_cons.add(item[1])
                else:
                    points_in_cons.add(item[1])
                    points_in_cons.add(item[2])
        elif cons.startswith('Collinear'):
            cons = cons.replace('Collinear(', '').replace(')', '')
            for item in cons:
                points_in_cons.add(item)
        elif cons.startswith('Cocircular'):
            cons = cons.replace('Cocircular(', '').replace(')', '').split(',')
            if len(cons) > 1:
                for item in cons[1]:
                    points_in_cons.add(item)

    return points_in_cons


def add_points_to_cdl():
    log_filename = '../../outputs/log/log_dataset_add_points_error.txt'
    log_add_points = []
    for pid in range(config['data']['max_pid']):
        pid += 1
        try:
            cdl = load_json(f'../../datasets/problems/{pid}.json')
            cdl['points'] = find_points_from_ggb(f'../../datasets/ggbs/{pid}.ggb')

            points_in_cons = get_points_in_cons(cdl['construction_cdl'])
            points_in_img = set(cdl['points'].keys())

            all_points = points_in_cons | points_in_img
            points_check = (all_points - points_in_cons) | (all_points - points_in_img)
            if len(points_check) > 0:
                points_in_cons = f"[{','.join(sorted(list(points_in_cons)))}]"
                points_in_img = f"[{','.join(sorted(list(points_in_img)))}]"
                points_check = f"[{','.join(sorted(list(points_check)))}]"
                msg = ', '.join([points_in_cons, points_in_img, points_check])
                log_add_points.append('\t'.join([str(pid), '0', msg]))
                print(log_add_points[-1])
            else:
                save_json(cdl, f'../../datasets/problems/{pid}.json')
                print(pid, 1, 'ok')
        except BaseException as e:
            log_add_points.append('\t'.join([str(pid), '2', repr(e)]))
            print(log_add_points[-1])

    with open(log_filename, 'w', encoding='utf-8') as file:
        file.write("\n".join(log_add_points))


def adjust_image(size=2048):
    log_filename = '../../outputs/log/log_dataset_adjust_image_error.txt'
    log_adjust_image = []
    for pid in range(config['data']['max_pid']):
        pid += 1
        try:
            image = cv2.imread(f'../../datasets/diagrams/{pid}.png')

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 将图像二值化
            coords = cv2.findNonZero(thresh)  # 返回非零像素的坐标
            x, y, w, h = cv2.boundingRect(coords)  # 计算边界框
            image = image[y:y + h, x:x + w]

            height, width, _ = image.shape
            max_side = int(max(height, width) * 1.05)  # 多5%的边界
            top_padding = (max_side - height) // 2  # 计算填充的边界
            bottom_padding = max_side - height - top_padding
            left_padding = (max_side - width) // 2
            right_padding = max_side - width - left_padding

            # 使用边界填充将图像调整为正方形
            image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                       cv2.BORDER_CONSTANT, value=[255, 255, 255])
            image = cv2.resize(image, (size, size))  # 调整图像大小

            cv2.imwrite(f'../../datasets/diagrams/{pid}.png', image)
            print(pid, 'ok')
        except BaseException as e:
            log_adjust_image.append('\t'.join([str(pid), 'error', repr(e)]))
            print(log_adjust_image[-1])

    with open(log_filename, 'w', encoding='utf-8') as file:
        file.write("\n".join(log_adjust_image))


def check_one_problem(problem_id, parsed_gdl):
    cdl = load_json(f'../../datasets/problems/{problem_id}.json')
    parsed_cdl = parse_cdl(cdl)
    problem_initial = Problem(parsed_gdl, parsed_cdl)

    problem = deepcopy(problem_initial)
    for theorem in cdl['theorem_seqs']:
        problem.apply(theorem)
    if problem.status_of_goal[0] != 1:
        raise Exception('Annotated theorem seqs can not solve problem.')

    cdl = load_json(f'../../datasets/problems/{problem_id}.json')
    cdl['theorem_seqs'] = get_cleaned_theorem_seqs(problem_initial, cdl['theorem_seqs'])
    cdl['forward_dag'] = get_forward_dag(problem_initial, cdl['theorem_seqs'])
    cdl['backward_dag'] = get_backward_dag(problem_initial, cdl['theorem_seqs'])


def multiprocess_check(task_queue, reply_queue):
    warnings.filterwarnings("ignore")
    parsed_gdl = parse_gdl(load_json('../../datasets/gdl.json'))
    while not task_queue.empty():
        problem_id = task_queue.get()

        timing = time.time()
        try:
            func_timeout(
                timeout=config['data']['timeout'],
                func=check_one_problem,
                args=(problem_id, parsed_gdl)
            )
            reply_queue.put((os.getpid(), problem_id, 'checked', 'checked', time.time() - timing))
        except FunctionTimedOut:
            reply_queue.put((os.getpid(), problem_id, 'timeout',
                             f"timeout after {config['data']['timeout']}s", time.time() - timing))
        except BaseException as e:
            reply_queue.put((os.getpid(), problem_id, 'error', repr(e), time.time() - timing))


def check_all_problems(solve_again=False):
    log_path = f"../../outputs/log/log_check_all_problems.json"
    log_check_all_problems = {"checked": {}, "timeout": {}, "error": {}}
    if os.path.exists(log_path):
        log_check_all_problems = load_json(log_path)

    task_queue = Queue()
    problem_ids = []
    if solve_again:  # clear unsolved, timeout, and error, run again
        log_check_all_problems["timeout"] = {}
        log_check_all_problems["error"] = {}

    for problem_id in range(config['data']['max_pid']):
        problem_id += 1
        if str(problem_id) in log_check_all_problems["checked"]:
            continue
        if str(problem_id) in log_check_all_problems["timeout"]:
            continue
        if str(problem_id) in log_check_all_problems["error"]:
            continue
        problem_ids.append(problem_id)

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
                target=multiprocess_check,
                args=(task_queue, reply_queue)
            )
            process.start()
            process_ids.append(process.pid)

        if not reply_queue.empty():  # directly calling .get() will block process
            process_id, problem_id, result, msg, timing = reply_queue.get()
            log_check_all_problems[result][problem_id] = msg
            count += 1
            print(f"({count}/{len(problem_ids)}) process_id={process_id}, problem_id={problem_id}, "
                  f"result={result}, timing={timing}")
            save_json(log_check_all_problems, log_path)


def add_natural_language():
    pass


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")
    parser.add_argument("--func", type=str, required=True,
                        choices=["add_points_to_cdl", "adjust_image", "check_all_problems", "add_natural_language"],
                        help="function that you want to run")
    parser.add_argument("--solve_again", action="store_true", default=False)

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == '__main__':
    """
    python tools.py --func kill --filename dataset.py
    
    python dataset.py --func add_points_to_cdl
    python dataset.py --func adjust_image
    python dataset.py --func check_all_problems --solve_again
    python dataset.py --func add_natural_language
    """
    args = get_args()
    if args.func == "add_points_to_cdl":
        add_points_to_cdl()
    elif args.func == "adjust_image":
        adjust_image()
    elif args.func == "check_all_problems":
        check_all_problems(args.solve_again)
    elif args.func == "add_natural_language":
        add_natural_language()
