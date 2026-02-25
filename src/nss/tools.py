import json
from sympy import symbols, sympify, log, atan2, pi
from pprint import pprint
import re
import string
import random
from graphviz import Digraph
import time
from copy import deepcopy
import pickle
import argparse
import psutil
import os


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filename):
    filename_bk = filename + '.bk'
    with open(filename_bk, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if os.path.exists(filename):
        os.remove(filename)
    os.rename(filename_bk, filename)


def show_json(dict_data):
    pprint(dict_data, sort_dicts=False, compact=True)
    print()


def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def debug_execute(func, debug_execute_args):
    timing = time.time()
    result = func(*debug_execute_args)
    msg = (f"func: {func.__name__}, args: {str(debug_execute_args)}, return: {str(result)}, "
           f"take: {round(time.time() - timing, 4)}s.")
    if isinstance(result, bool):
        if result:
            print(f"\033[32m{msg}\033[0m")
        else:
            print(f"\033[31m{msg}\033[0m")
    else:
        print(msg)


config = load_json('../../config.json')


def parse_fact(s):
    """
    Parse s to get predicate name and paras.
    >> parse_geo_predicate('Predicate(A,B,C)')
    ('Predicate', ['A', 'B', 'C'])
    """
    predicate_name, paras = s.split("(")
    paras = paras[:-1].replace(",", "")
    return predicate_name, tuple(paras)


def parse_expr(s):
    """
    Parse str expression to sympy expression.
    Args:
        s (str): Algebra relation and expression. The components include algebra relation types,
        algebraic operations, the symbolic representations of measures and constants. Such as:
        'Eq(Sub(A.y,Add(Mul(l.k,A.x),l.b)))', 'Value(Mul(Sub(C.x,B.x),Sub(A.y,B.y)))'.

    Returns:
        parsed_s (tuple): Algebra relation type and instance of sympy expression. Such as:
        ('Eq', -A.x*l.k + A.y - l.b), ('Value', (A.y - B.y)*(-B.x + C.x)).
    """
    predicate, expr_str = s.split("(", 1)
    expr_str = expr_str[:-1]

    if '(' not in expr_str:  # such as 'Eq(lk.ma)'
        return predicate, symbols(expr_str)

    i = 0
    j = 0
    stack = []
    while j < len(expr_str):
        if expr_str[j] == "(":
            stack.append(expr_str[i:j])
            stack.append(expr_str[j])
            i = j + 1
        elif expr_str[j] == ",":
            if i < j:
                stack.append(expr_str[i: j])
                i = j + 1
            else:
                i = i + 1
        elif expr_str[j] == ")":
            if i < j:
                stack.append(expr_str[i: j])
                i = j + 1
            else:
                i = i + 1

            paras = []
            while True:
                para = stack.pop()
                if para == "(":
                    break
                if type(para) is str:
                    if '.' in para:
                        para = symbols(para)  # symbol representation of measure
                    else:
                        para = sympify(para.replace('{', '(').replace('}', ')'))  # constant, free symbols, or expr
                paras.append(para)
            paras = paras[::-1]

            operation = stack.pop()

            if operation == 'Add':
                result = paras[0]
                for p in paras[1:]:
                    result += p
            elif operation == 'Sub':
                result = paras[0] - paras[1]
            elif operation == 'Mul':
                result = paras[0]
                for p in paras[1:]:
                    result *= p
            elif operation == 'Div':
                result = paras[0] / paras[1]
            elif operation == 'Pow':
                result = paras[0] ** paras[1]
            elif operation == 'Log':
                result = log(paras[0])
            elif operation == 'Ma':
                a_x, a_y, b_x, b_y, c_x, c_y = paras
                BA = (a_x - b_x, a_y - b_x)  # vector BA
                BC = (c_x - b_x, c_y - b_y)  # vector BC
                angle_BA = atan2(BA[1], BA[0])  # (-π, π]
                angle_BC = atan2(BC[1], BC[0])  # (-π, π]
                angle_diff = angle_BC - angle_BA
                result = (2 * pi - angle_diff) % (2 * pi)  # clockwise
            else:
                e_msg = f"Unknown operation '{operation}' in s '{s}'."
                raise Exception(e_msg)

            stack.append(result)

        j = j + 1

    if len(stack) > 1:
        e_msg = f"Syntax error in s '{s}': missing ')'?"
        raise Exception(e_msg)

    return predicate, stack.pop()


def replace_paras(paras, replace):
    replaced_paras = [replace[p] for p in paras]
    return tuple(replaced_paras)


def replace_expr(expr, replace):
    """Replace instances according to the replacement mapping.

    Args:
        expr (sympy_expr): instance of sympy expression. Such as -A.x*l.k + A.y - l.b.
        replace (dict): Keys are the old entity and values are the new entity. Such As {'A': 'B', 'l': 'k'}.

    Returns:
        replaced_expr: Replaced expr. Such as -B.x*k.k + B.y - k.b.
    """
    replace_old_to_temp = {}
    replace_temp_to_new = {}
    for sym_old in expr.free_symbols:
        entities_old, attr = str(sym_old).split('.')

        sym_temp = symbols("".join([e + "'" for e in entities_old]) + '.' + attr)
        replace_old_to_temp[sym_old] = sym_temp

        sym_new = symbols("".join([replace[e] for e in entities_old]) + '.' + attr)
        replace_temp_to_new[sym_temp] = sym_new

    expr = expr.subs(replace_old_to_temp).subs(replace_temp_to_new)

    return expr


def parse_disjunctive(s):
    if len(s) == 0:
        return []
    return s.split('&')


def parse_gdl(gdl):
    parsed_gdl = {
        'Presets': {},
        'Relations': {},
        'Attributions': {},
        'sym_to_attr': {},
        'Theorems': {}
    }

    for preset in gdl['Presets']:
        preset_name, preset_paras = parse_fact(preset)
        parsed_gdl['Presets'][preset_name] = {
            'paras': preset_paras
        }

    for relation in gdl['Relations']:
        relation_name, relation_paras = parse_fact(relation)
        geometric_constraints = []
        for geometric_constraint in parse_disjunctive(gdl['Relations'][relation]['geometric_constraints']):
            name, paras = parse_fact(geometric_constraint)
            geometric_constraints.append((name, paras))
        parsed_gdl['Relations'][relation_name] = {
            'paras': relation_paras,
            'geometric_constraints': tuple(geometric_constraints)
        }

    for attr in gdl['Attributions']:
        attr_name, attr_paras = parse_fact(attr)
        geometric_constraints = []
        for geometric_constraint in parse_disjunctive(gdl['Attributions'][attr]['geometric_constraints']):
            name, paras = parse_fact(geometric_constraint)
            geometric_constraints.append((name, paras))
        multiple_forms = []
        for multi in parse_disjunctive(gdl['Attributions'][attr]['multiple_forms']):
            _, multi_paras = parse_fact(multi)
            multiple_forms.append(multi_paras)
        parsed_gdl['Attributions'][gdl['Attributions'][attr]['sym']] = {
            'name': attr_name,
            'paras': attr_paras,
            'geometric_constraints': tuple(geometric_constraints),
            'multiple_forms': tuple(multiple_forms)
        }

    for theorem in gdl['Theorems']:
        _parse_one_theorem(theorem, gdl, parsed_gdl)

    return parsed_gdl


def _parse_one_theorem(theorem, gdl, parsed_gdl):
    theorem_name, theorem_paras = parse_fact(theorem)

    geometric_constraints = []  # (predicate, paras)
    geometric_premises = []  # (predicate, paras)
    algebraic_premises = []  # (expr, paras)
    algebraic_constraints = []  # (relation_type, expr, paras)

    for premise in parse_disjunctive(gdl['Theorems'][theorem]['premises']):
        if premise.startswith('Eq('):
            _, expr = parse_expr(premise)
            paras = []
            for sym in expr.free_symbols:
                paras.extend(list(str(sym).split('.')[0]))
            algebraic_premises.append((expr, paras))
        else:
            premise_name, premise_paras = parse_fact(premise)
            geometric_premises.append((premise_name, premise_paras))

            if premise_name in parsed_gdl['Presets']:
                geometric_constraints.append((premise_name, premise_paras))
            else:
                replace = dict(zip(parsed_gdl['Relations'][premise_name]['paras'], premise_paras))
                for predicate, paras in parsed_gdl['Relations'][premise_name]['geometric_constraints']:
                    paras = replace_paras(paras, replace)
                    geometric_constraints.append((predicate, paras))

    for constraint in parse_disjunctive(gdl['Theorems'][theorem]['algebraic_constraints']):
        algebra_relation, expr = parse_expr(constraint)
        paras = [str(sym).split('.')[0] for sym in expr.free_symbols]
        algebraic_constraints.append((algebra_relation, expr, paras))

    entities_gpl = _get_gpl(geometric_constraints, [], algebraic_constraints, theorem_paras)
    premises_gpl = _get_gpl(geometric_premises, algebraic_premises, algebraic_constraints, theorem_paras)

    # parse theorem conclusions
    if gdl['Theorems'][theorem]['conclusion'].startswith('Eq('):
        _, expr = parse_expr(gdl['Theorems'][theorem]['conclusion'])
        conclusion = ('Eq', expr)
    else:
        conclusion_name, conclusion_paras = parse_fact(gdl['Theorems'][theorem]['conclusion'])
        conclusion = (conclusion_name, conclusion_paras)
    # print(gdl['Theorems'][theorem])
    parsed_gdl['Theorems'][theorem_name] = {
        'paras': theorem_paras,
        'circle': set(gdl['Theorems'][theorem]['circle']),
        'entities_gpl': entities_gpl,
        'premises_gpl': premises_gpl,
        'conclusion': conclusion
    }


def _get_gpl(geometric_premises, algebraic_premises, algebraic_constraints, theorem_paras):
    geometric_premises = list(geometric_premises)  # (predicate, paras)
    algebraic_premises = list(algebraic_premises)  # (expr, paras)
    algebraic_constraints = list(algebraic_constraints)  # (relation_type, expr, paras)

    # adjust the execution order
    products = []
    added_paras = set()

    # map para to geometric_premises
    paras_to_geometric_premises = {}
    for premise_name, premise_paras in geometric_premises:
        for p in list(set(premise_paras)):
            if p not in paras_to_geometric_premises:
                paras_to_geometric_premises[p] = [(premise_name, premise_paras)]
            else:
                paras_to_geometric_premises[p].append((premise_name, premise_paras))

    # add geometric_premise to product, entity p only exist in those geometric_premise
    for p in paras_to_geometric_premises:
        if len(paras_to_geometric_premises[p]) == 1 and paras_to_geometric_premises[p][0] not in products:
            products.append(paras_to_geometric_premises[p][0])
            geometric_premises.remove(paras_to_geometric_premises[p][0])
            added_paras.update(paras_to_geometric_premises[p][0][1])

    # for the remaining geometric_premise, select a portion to add to product, according to:
    # 1. the number of not added entities in it paras
    # 2. the number of paras
    # print(products)
    # print(paras_to_geometric_premises)
    # print(added_paras)
    # print()
    while len(added_paras) < len(theorem_paras):
        # print(added_paras)
        # print(theorem_paras)
        # print(theorem_geometric_premises)
        max_index = 0
        max_not_added_paras_len = len(set(geometric_premises[0][1]) - added_paras)
        max_paras_len = len(geometric_premises[0][1])

        for i in range(1, len(geometric_premises)):
            not_added_paras_len = len(set(geometric_premises[i][1]) - added_paras)
            paras_len = len(geometric_premises[i][1])

            if not_added_paras_len > max_not_added_paras_len or (
                    not_added_paras_len == max_not_added_paras_len and paras_len > max_paras_len):
                max_index = i
                max_not_added_paras_len = not_added_paras_len
                max_paras_len = paras_len

        products.append(geometric_premises[max_index])
        added_paras.update(geometric_premises[max_index][1])
        geometric_premises.pop(max_index)

    # sort product according to the number of its paras
    products.sort(key=len, reverse=True)

    gpl = []
    added_paras = []
    for predicate, paras in products:
        inherent_same_index = []
        for i in range(len(paras)):
            for j in range(i + 1, len(paras)):
                if paras[i] == paras[j]:
                    inherent_same_index.append((i, j))
        mutual_same_index = []
        for i in range(len(added_paras)):
            for j in range(len(paras)):
                if added_paras[i] == paras[j]:
                    mutual_same_index.append((i, j))
        added_index = []
        for j in range(len(paras)):
            if paras[j] not in added_paras:
                added_index.append(j)
                added_paras.append(paras[j])

        geometric_premise = _get_geometric_premise(geometric_premises, added_paras)  # (predicate, paras)
        algebraic_premise = _get_algebraic_premise(algebraic_premises, added_paras)  # (expr)
        algebraic_constraint = _get_algebraic_constraint(algebraic_constraints, added_paras)  # (relation_type, expr)

        gpl.append({
            "product": (predicate, paras, tuple(inherent_same_index), tuple(mutual_same_index), tuple(added_index)),
            "geometric_premises": geometric_premise,
            "algebraic_premises": algebraic_premise,
            "algebraic_constraints": algebraic_constraint
        })

    if len(geometric_premises) > 0 or len(algebraic_premises) > 0 or len(algebraic_constraints) > 0:
        e_msg = f"There exist unadded constraints."
        raise Exception(e_msg)

    return tuple(gpl)


def _get_algebraic_constraint(algebraic_constraints, added_paras):
    algebraic_constraint = []  # (relation_type, expr, paras)
    for i in range(len(algebraic_constraints))[::-1]:
        ac_check_type, ac_check_expr, ac_check_paras = algebraic_constraints[i]
        if len(set(ac_check_paras) - set(added_paras)) == 0:
            algebraic_constraint.append(algebraic_constraints[i])
            algebraic_constraints.pop(i)
    # sort according to the number of paras
    algebraic_constraint = sorted(algebraic_constraint, key=lambda x: (len(x[2]), len(set(x[2]))), reverse=True)
    algebraic_constraint = tuple([(relation_type, expr) for relation_type, expr, _ in algebraic_constraint])
    return algebraic_constraint


def _get_geometric_premise(geometric_premises, added_paras):
    geometric_premise = []  # (predicate, paras)
    for i in range(len(geometric_premises))[::-1]:
        geometric_premises_predicate, geometric_premises_paras = geometric_premises[i]
        if len(set(geometric_premises_paras) - set(added_paras)) == 0:
            geometric_premise.append(geometric_premises[i])
            geometric_premises.pop(i)
    # sort according to the number of paras
    geometric_premise = tuple(sorted(geometric_premise, key=lambda x: (len(x[1]), len(set(x[1]))), reverse=True))
    return geometric_premise


def _get_algebraic_premise(algebraic_premises, added_paras):
    algebraic_premise = []  # (expr, paras)
    for i in range(len(algebraic_premises))[::-1]:
        algebraic_premises_expr, algebraic_premises_paras = algebraic_premises[i]
        if len(set(algebraic_premises_paras) - set(added_paras)) == 0:
            algebraic_premise.append(algebraic_premises[i])
            algebraic_premises.pop(i)
    algebraic_premise = sorted(algebraic_premise, key=lambda x: (len(x[1]), len(set(x[1]))), reverse=True)
    algebraic_premise = tuple([expr for expr, _ in algebraic_premise])
    return algebraic_premise


def parse_cdl(cdl):
    construction_cdl = []
    for one_cdl in cdl['construction_cdl']:
        if one_cdl.startswith("Shape"):
            predicate, paras = one_cdl.split('(')
            paras = tuple(paras[:-1].split(','))
        elif one_cdl.startswith('Collinear'):
            predicate, paras = one_cdl.split('(')
            paras = tuple(paras[:-1])
        else:
            predicate, paras = one_cdl.split('(')
            paras = tuple(paras[:-1].replace(',', ''))
        construction_cdl.append((predicate, paras))

    points = {}
    for point in cdl['points']:
        points[point] = tuple(cdl['points'][point])

    relation_cdl = []
    for one_cdl in cdl['text_cdl'] + cdl['image_cdl']:
        if one_cdl.startswith('Eq('):
            fact = parse_expr(one_cdl)
        else:
            fact = parse_fact(one_cdl)

        if fact not in relation_cdl:
            relation_cdl.append(fact)

    if cdl['goal_cdl'].startswith('Eq('):
        goal_cdl = parse_expr(cdl['goal_cdl'])
    else:
        goal_cdl = parse_fact(cdl['goal_cdl'])

    parsed_cdl = {
        'problem_id': cdl['problem_id'],
        'construction_cdl': tuple(construction_cdl),
        'points': points,
        'relation_cdl': tuple(relation_cdl),
        'goal_cdl': goal_cdl
    }

    # for predicate, instance in parsed_cdl['relation_cdl']:
    #     if predicate == 'Eq':
    #         for sym in instance.free_symbols:
    #             print(f'{str(sym)}: ', sym == symbols(str(sym)))

    return parsed_cdl


def get_used_theorems():
    used_theorems = set()
    for pid in range(config['data']['max_pid']):
        pid += 1
        for theorem in load_json(f'../../datasets/problems/{pid}.json')['theorem_seqs']:
            used_theorems.add(theorem.split('(')[0])

    return sorted(list(used_theorems))


expr_letters = tuple(  # letters in algebraic expr
    ['+', '-', '**', '*', '/', 'sqrt', 'number', 'pi', '(', ')'] +
    sorted(['.' + attr_sym for attr_sym in parse_gdl(load_json('../../datasets/gdl.json'))['Attributions'].keys()])
)

theorem_letters = tuple(  # theorem letters (theorem vocab)
    ['solve_eq'] + get_used_theorems()
    # sorted(list(parse_gdl(load_json('../../datasets/gdl.json'))['Theorems'].keys()))
)

state_letters = tuple(  # letters in serialized problem state
    ['padding'] +
    list(expr_letters) +  # letters in algebraic expr
    [  # delimiter letter
        ',', '&', '|',  # split facts
        '<construction>',  # construction
        '<init_fact>', '<premise>', '<apply_theorem>', '<conclusion>',  # forward
        '<init_goal>', '<goal>', '<decompose>', '<sub_goals>'  # backward
    ] +
    sorted([r for r in parse_gdl(load_json('../../datasets/gdl.json'))['Presets'].keys()]) +  # Predicate
    sorted([r for r in parse_gdl(load_json('../../datasets/gdl.json'))['Relations'].keys()]) +  # Predicate
    list(string.ascii_letters) +  # parameters
    list(theorem_letters)  # # theorem letters (theorem vocab)
)


def _anti_parse_operation(operation):
    operation_type, operation_predicate, operation_instance = operation
    if operation_type == 'Preset':
        return 'Preset: ' + operation_predicate
    elif operation_type == 'Apply':
        return 'Apply: ' + operation_predicate + '(' + ','.join(operation_instance) + ')'
    elif operation_type == 'Decompose':
        return 'Decompose: ' + operation_predicate + '(' + ','.join(operation_instance) + ')'
    else:
        raise Exception(f"Unknown operation type '{operation_type}'.")


def _serialize_fact(predicate, instance):
    if predicate == 'Eq':
        # print(instance)
        serialized_expr = ['Eq']
        expr = str(instance).replace(' ', '')  # remove ' '

        for matched in re.findall(r'\d+\.*\d*', expr):  # replace number with 'nums'
            expr = expr.replace(matched, 'number', 1)

        i = 0
        while i < len(expr):  # serialize
            added = False
            for matched_part in expr_letters:  # expr letters
                if expr[i:].startswith(matched_part):
                    serialized_expr.append(matched_part)
                    i = i + len(matched_part)
                    added = True
                    break
            if not added:  # entity letters
                serialized_expr.append(expr[i])
                i = i + 1
        # print(serialized_expr)
        # print()
        return serialized_expr
    else:
        return [predicate] + list(instance)


def _serialize_operation(operation):
    operation_type, operation_predicate, operation_instance = operation
    if operation_type == 'Preset':
        return [operation_predicate]
    elif operation_type == 'Apply':
        return [operation_predicate] + list(operation_instance)
    elif operation_type == 'Decompose':
        return [operation_predicate] + list(operation_instance)
    else:
        raise Exception(f"Unknown operation type '{operation_type}'.")


precision = 15
chop = 1e-10


def _satisfy_eq(expr, sym_to_value=None):
    try:
        if sym_to_value is None:
            return expr.evalf(n=precision, chop=chop) == 0
        return expr.subs(sym_to_value).evalf(n=precision, chop=chop) == 0
    except Exception:
        return False


def _satisfy_g(expr, sym_to_value=None):
    try:
        if sym_to_value is None:
            return expr.evalf(n=precision, chop=chop) > 0
        return expr.subs(sym_to_value).evalf(n=precision, chop=chop) > 0
    except Exception:
        return False


def _satisfy_geq(expr, sym_to_value=None):
    try:
        if sym_to_value is None:
            return expr.evalf(n=precision, chop=chop) >= 0
        return expr.subs(sym_to_value).evalf(n=precision, chop=chop) >= 0
    except Exception:
        return False


def _satisfy_l(expr, sym_to_value=None):
    try:
        if sym_to_value is None:
            return expr.evalf(n=precision, chop=chop) < 0
        return expr.subs(sym_to_value).evalf(n=precision, chop=chop) < 0
    except Exception:
        return False


def _satisfy_leq(expr, sym_to_value=None):
    try:
        if sym_to_value is None:
            return expr.evalf(n=precision, chop=chop) <= 0
        # print('Leq')
        # print(expr)
        # print(sym_to_value)
        # print(expr.subs(sym_to_value))
        # print(expr.subs(sym_to_value).evalf(n=precision, chop=chop))
        # print((expr / pi * 180).subs(sym_to_value).evalf(n=precision, chop=chop))
        # print(expr.subs(sym_to_value).evalf(n=precision, chop=chop) <= 0)
        # print()
        return expr.subs(sym_to_value).evalf(n=precision, chop=chop) <= 0
    except Exception:
        return False


def _satisfy_ueq(expr, sym_to_value=None):
    try:
        if sym_to_value is None:
            return expr.evalf(n=precision, chop=chop) != 0
        return expr.subs(sym_to_value).evalf(n=precision, chop=chop) != 0
    except Exception:
        return False


_satisfy_algebraic = {'Eq': _satisfy_eq, 'G': _satisfy_g, 'Geq': _satisfy_geq,
                      'L': _satisfy_l, 'Leq': _satisfy_leq, 'Ueq': _satisfy_ueq}


def get_theorem_seqs(problem):
    theorem_seqs = []
    goal_related_premise_ids = list(problem.premise_ids_of_goal[0])
    goal_related_operation_ids = set()
    for fact_id in goal_related_premise_ids:
        goal_related_operation_ids.add(problem.facts[fact_id][3])
        for new_fact_id in problem.facts[fact_id][2]:
            if new_fact_id not in goal_related_premise_ids:
                goal_related_premise_ids.append(new_fact_id)

    for operation_id in range(len(problem.operations)):
        if operation_id not in goal_related_operation_ids:
            continue
        operation_type, operation_predicate, operation_instance = problem.operations[operation_id]
        if operation_type != 'Apply':
            continue

        theorem_seqs.append(operation_predicate + '(' + ','.join(operation_instance) + ')')

    return theorem_seqs


def get_cleaned_theorem_seqs(problem_initial, theorem_seqs):
    theorem_seqs = deepcopy(theorem_seqs)
    for i in range(len(theorem_seqs))[::-1]:  # try delete theorem i
        problem = deepcopy(problem_initial)
        for j in range(len(theorem_seqs)):  # not apply theorem i
            if j == i:
                continue
            problem.apply(theorem_seqs[j])

        if problem.status_of_goal[0] == 1:  # theorem i can delete
            theorem_seqs.pop(i)

    return theorem_seqs


def get_dag(applied_theorems, edges):
    n = len(applied_theorems)  # 这里去重的代码，还能再优化下
    closure = [[False] * n for _ in range(n)]

    for head, tail in edges:
        closure[applied_theorems.index(head)][applied_theorems.index(tail)] = True

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if closure[i][k] and closure[k][j]:
                    closure[i][j] = True

    for i in range(n):
        for j in range(n):
            if closure[i][j]:
                for k in range(n):
                    if k != i and k != j and closure[i][k] and closure[k][j]:
                        if (applied_theorems[i], applied_theorems[j]) in edges:
                            edges.remove((applied_theorems[i], applied_theorems[j]))
                        break
    dag = {
        'in_degree': {},
        'out_degree': {},
        'edges': []
    }
    for theorem in applied_theorems:
        dag['in_degree'][theorem] = 0
        dag['out_degree'][theorem] = 0
    for head, tail in edges:
        dag['in_degree'][tail] += 1
        dag['out_degree'][head] += 1
    dag['edges'] = edges

    return dag


def get_forward_dag(problem_initial, theorem_seqs):
    theorem_seqs = deepcopy(theorem_seqs)
    previous_problem = deepcopy(problem_initial)
    applied_theorems = []
    edges = []
    while len(theorem_seqs) > 0:
        for i in range(len(theorem_seqs))[::-1]:
            problem = deepcopy(previous_problem)
            if not problem.apply(theorem_seqs[i]):  # check whether theorem i can apply under previous theorems
                continue

            dependent_theorems = deepcopy(applied_theorems)
            for j in range(len(dependent_theorems))[::-1]:  # check whether theorem j is dependent
                problem = deepcopy(problem_initial)
                for k in range(len(dependent_theorems)):  # not apply theorem k=j
                    if k == j:
                        continue
                    problem.apply(dependent_theorems[k])

                if problem.apply(theorem_seqs[i]):  # still can apply theorem i after delete theorem j
                    dependent_theorems.pop(j)

            check_theorem = theorem_seqs.pop(i)
            applied_theorems.append(check_theorem)
            previous_problem.apply(check_theorem)
            for dependent_theorem in dependent_theorems:
                edges.append((dependent_theorem, check_theorem))

    return get_dag(applied_theorems, edges)


def get_backward_dag(problem_initial, theorem_seqs):
    theorem_seqs = deepcopy(theorem_seqs)
    previous_problem = deepcopy(problem_initial)
    applied_theorems = []
    edges = []
    while len(theorem_seqs) > 0:
        for i in range(len(theorem_seqs))[::-1]:
            problem = deepcopy(previous_problem)
            if not problem.decompose(theorem_seqs[i]):  # check whether theorem i can apply under previous theorems
                continue

            dependent_theorems = deepcopy(applied_theorems)
            for j in range(len(dependent_theorems))[::-1]:  # check whether theorem j is dependent
                problem = deepcopy(problem_initial)
                for k in range(len(dependent_theorems)):  # not apply theorem k=j
                    if k == j:
                        continue
                    problem.decompose(dependent_theorems[k])

                if problem.decompose(theorem_seqs[i]):  # still can apply theorem i after delete theorem j
                    dependent_theorems.pop(j)

            check_theorem = theorem_seqs.pop(i)
            applied_theorems.append(check_theorem)
            previous_problem.decompose(check_theorem)
            for dependent_theorem in dependent_theorems:
                edges.append((dependent_theorem, check_theorem))

    return get_dag(applied_theorems, edges)


def inverse_parse_theorem(theorem):
    operation_type, operation_predicate, operation_instance = theorem
    if operation_type == 'Preset':
        return operation_predicate
    else:
        return operation_predicate + '(' + ','.join(operation_instance) + ')'


def inverse_parse_cdl(predicate, instance):
    if predicate in _satisfy_algebraic.keys():
        return predicate + '(' + str(instance).replace(' ', '') + ')'
    else:
        return predicate + '(' + ','.join(instance) + ')'


def get_meta_hypertree(problem):
    """
    Generate meta hypertree message for downstream task.
    :return nodes: all nodes, {node_id: node_name}, such as {1: 'Equation(ll_ab-1)'}
    :return edges: all edges, {edge_id: edge_name}, such as {1: "extended"}
    :return free_nodes: nodes not in hypertree but in prerequisite, [node_id], such as [1, 2, 3]
    :return target_node_id: target node id, such as 1
    :return hypertree: {((tail_node_ids), edge_id): (tail_node_ids))}, such as {((1, 2, 3), 1): (4, 5))}
    """
    group = {}  # (premise, theorem): [_id], used for building hyper graph.
    cdl = {}  # _id: anti_parsed_cdl, user for getting cdl by id.
    init_nodes = []  # [_id], id of prerequisite.
    tree_nodes = []  # [_id], id of tree nodes.
    target_node_id = None

    for fact_id in range(len(problem.facts)):
        predicate, instance, premise_ids, operation_id = problem.facts[fact_id]
        premise_ids = tuple(sorted(list(premise_ids)))
        theorem = inverse_parse_theorem(problem.operations[operation_id])

        if theorem == "extend_construction":  # 不需要这些节点
            continue

        cdl[fact_id] = inverse_parse_cdl(predicate, instance)

        if theorem in {'init_construction', 'init_fact'}:  # root nodes
            init_nodes.append(fact_id)
            continue

        if (premise_ids, theorem) not in group:
            group[(premise_ids, theorem)] = [fact_id]
        else:
            group[(premise_ids, theorem)].append(fact_id)

    if len(problem.goals) > 0 and problem.status_of_goal[0] == 1:
        predicate, instance, _, _ = problem.goals[0]
        if predicate == 'Eq' and (predicate, instance) not in problem.fact_id:
            target_node_id = len(problem.facts)
            cdl[target_node_id] = predicate + '(' + str(instance).replace(' ', '') + ')'
            premise_ids = tuple(sorted(list(problem.premise_ids_of_goal[0])))
            group[(premise_ids, 'solve_eq')] = [target_node_id]
        else:
            target_node_id = problem.fact_id[(predicate, instance)]

    # for cdl_key in cdl.keys():
    #     print(cdl_key, cdl[cdl_key])
    # print()
    #
    # for group_key in group:
    #     print(group_key, group[group_key])
    # print()

    edges = {-2: "none", -1: "self"}
    tree = {}
    for premise, theorem in group:
        conclusion = group[(premise, theorem)]
        edge_id = len(edges)
        edges[edge_id] = theorem

        adjust_premise = []
        for fact_id in premise:
            if fact_id in cdl:
                adjust_premise.append(fact_id)
            else:
                _, _, premise_ids, _ = problem.facts[fact_id]
                adjust_premise.extend(premise_ids)
        adjust_premise = sorted(list(set(adjust_premise)))

        tree_nodes += adjust_premise
        tree_nodes += conclusion
        tree[(tuple(adjust_premise), edge_id)] = conclusion

    nodes = {}
    for node_id in sorted(list(set(tree_nodes + init_nodes))):
        nodes[node_id] = cdl[node_id]

    free_nodes = sorted(list(set(init_nodes) - set(tree_nodes)))

    return nodes, edges, free_nodes, target_node_id, tree


def get_solution_hypertree(problem):
    """Generate solution hyper tree."""
    nodes, edges, free_nodes, target_node_id, tree = get_meta_hypertree(problem)
    parsed_tree = {}
    for premise, theorem in tree:
        conditions = [nodes[node_id] for node_id in premise]
        conclusions = [nodes[node_id] for node_id in tree[(premise, theorem)]]
        theorem = edges[theorem]
        parsed_tree[len(parsed_tree) + 1] = {
            "conditions": conditions,
            "theorem": theorem,
            "conclusions": conclusions
        }

    hypertree = {
        "nodes": list(nodes.values()),
        "free_nodes": [nodes[node_id] for node_id in free_nodes],
        "target_node": nodes[target_node_id] if target_node_id is not None else "None",
        "tree": parsed_tree
    }
    return hypertree


def draw_solution_hypertree(problem, filename):
    """Draw solution hyper tree and save as .png."""
    hypertree = get_solution_hypertree(problem)
    dot = Digraph()

    for node in hypertree["nodes"]:
        if node not in hypertree['free_nodes']:
            dot.node(node, node, shape="box", style='filled', fillcolor='lightyellow')
        else:
            dot.node(node, node, shape="box", style='filled', fillcolor='lightgrey')

    edges_count = 1
    for tree_id in hypertree["tree"]:
        dot.node(str(edges_count), hypertree["tree"][tree_id]["theorem"], style='filled', fillcolor='lightblue')

        for condition in hypertree["tree"][tree_id]["conditions"]:
            dot.edge(condition, str(edges_count))

        for conclusion in hypertree["tree"][tree_id]["conclusions"]:
            dot.edge(str(edges_count), conclusion)
        edges_count += 1

    dot.render(filename=filename, view=False, format="pdf", cleanup=True)  # save theorem DAG


def draw_dag(dag, filename):
    dot = Digraph()
    dot.attr(rankdir='TB')

    nodes = set()
    for node in dag['in_degree']:
        nodes.add(node)
    for node in dag['out_degree']:
        nodes.add(node)
    for node in sorted(nodes):
        dot.node(node, node, style='filled', fillcolor='lightblue')
    dot.node('START', 'START', style='filled', fillcolor='lightgreen', shape='box')
    dot.node('END', 'END', style='filled', fillcolor='lightgreen', shape='box')

    for src, dst in dag['edges']:
        dot.edge(src, dst)

    for node in dag['in_degree']:
        if dag['in_degree'][node] == 0:
            dot.edge('START', node)

    for node in dag['out_degree']:
        if dag['out_degree'][node] == 0:
            dot.edge(node, 'END', )

    dot.render(filename=filename, view=False, cleanup=True, format='pdf')


def delete_node_from_dag(node, dag):
    node_as_tail = []
    node_as_head = []
    for head, tail in dag['edges']:
        if head == node:
            node_as_head.append((head, tail))
        if tail == node:
            node_as_tail.append((head, tail))

    for head, tail in node_as_tail + node_as_head:
        dag['edges'].remove((head, tail))
        dag['out_degree'][head] -= 1
        dag['in_degree'][tail] -= 1

    for a_head, a_tail in node_as_tail:
        for b_head, b_tail in node_as_head:
            if (a_head, b_tail) in dag['edges']:
                continue
            dag['edges'].append((a_head, b_tail))
            dag['out_degree'][a_head] += 1
            dag['in_degree'][b_tail] += 1


def topological_sort(dag, random_seed):
    dag = deepcopy(dag)
    rng = random.Random(random_seed)

    theorem_seqs = []
    candidates = [node for node in dag['in_degree']
                  if dag['in_degree'][node] == 0 and node not in theorem_seqs]
    while len(candidates) > 0:
        theorem = rng.choice(candidates)
        theorem_seqs.append(theorem)

        delete_node_from_dag(theorem, dag)

        candidates = [node for node in dag['in_degree']
                      if dag['in_degree'][node] == 0 and node not in theorem_seqs]
    return theorem_seqs


def topological_sort_bidirectional(forward_dag, backward_dag, random_seed):
    forward_dag = deepcopy(forward_dag)
    backward_dag = deepcopy(backward_dag)
    if set(forward_dag['in_degree'].keys()) != set(backward_dag['in_degree'].keys()):
        raise Exception('forward_dag and backward_dag asymmetry!')

    rng = random.Random(random_seed)

    directions = []
    theorem_seqs = []
    forward_candidates = [node for node in forward_dag['in_degree']
                          if forward_dag['in_degree'][node] == 0 and node not in theorem_seqs]
    backward_candidates = [node for node in backward_dag['in_degree']
                           if backward_dag['in_degree'][node] == 0 and node not in theorem_seqs]
    while len(forward_candidates) > 0 or len(backward_candidates) > 0:
        # show_json(forward_dag)
        # show_json(backward_dag)
        # print(theorem_seqs)
        # print()
        # print()
        # print()
        if rng.random() < 0.5:
            theorem = rng.choice(forward_candidates)
            directions.append('f')
            theorem_seqs.append(theorem)
        else:
            theorem = rng.choice(backward_candidates)
            directions.append('b')
            theorem_seqs.append(theorem)

        delete_node_from_dag(theorem, forward_dag)
        delete_node_from_dag(theorem, backward_dag)

        forward_candidates = [node for node in forward_dag['in_degree']
                              if forward_dag['in_degree'][node] == 0 and node not in theorem_seqs]
        backward_candidates = [node for node in backward_dag['in_degree']
                               if backward_dag['in_degree'][node] == 0 and node not in theorem_seqs]
    return list(zip(directions, theorem_seqs))


def kill(filename):
    for pid in psutil.pids():
        process = psutil.Process(pid)
        if not (process.name() == "python" and filename in process.cmdline()):
            continue

        try:
            process.kill()
            print(f"kill process {pid}")
        except BaseException as e:
            print(f'Error when kill {pid}:', repr(e))


def get_acc(level=6, span=3, latex=False):
    filenames = {
        'T5-small': 'log_pssr_t5-small_bs20_timeout600.json',
        'BART-base': 'log_pssr_bart-base_bs20_timeout600.json',

        'Inter-GPS': 'log_pssr_intergps.json',
        'NGS': 'log_pssr_ngs_bs10_timeout600.json',
        'DualGeoSolver': 'log_pssr_dualgeosolver_bs10_timeout600.json',

        'FGeo-DRL': 'log_pssr_fgeodrl.json',
        'FGeo-HyperGNet': 'log_pssr_hypergnet_TTT_bs5_gb_tm600.json',

        'Forward-BFS': 'log_pssr_formalgeo7k-fw-bfs.json',
        'Forward-DFS': 'log_pssr_formalgeo7k-fw-dfs.json',
        'Forward-RS': 'log_pssr_formalgeo7k-fw-rs.json',
        'Backward-BFS': 'log_pssr_formalgeo7k-bw-bfs.json',
        'Backward-DFS': 'log_pssr_formalgeo7k-bw-dfs.json',
        'Backward-RS': 'log_pssr_formalgeo7k-bw-rs.json',

        'DeepSeek v3': 'log_pssr_deepseek-v3.json',
        'Kimi-K2': 'log_pssr_kimi-k2.json',

        'Ours': 'log_pssr_nss_FFFF_bs5_tm600.json',

        'Text-only-1': 'log_pssr_nss_TFFF_bs1_tm60.json',
        'Text-only-3': 'log_pssr_nss_TFFF_bs3_tm60.json',
        'Text-only-5': 'log_pssr_nss_TFFF_bs5_tm60.json',

        'Forward-only-1': 'log_pssr_nss_FTFF_bs1_tm60.json',
        'Forward-only-3': 'log_pssr_nss_FTFF_bs3_tm60.json',
        'Forward-only-5': 'log_pssr_nss_FTFF_bs5_tm60.json',

        'No Gate-1': 'log_pssr_nss_FFTF_bs1_tm60.json',
        'No Gate-3': 'log_pssr_nss_FFTF_bs3_tm60.json',
        'No Gate-5': 'log_pssr_nss_FFTF_bs5_tm60.json',

        'Small Model-1': 'log_pssr_nss_FFFT_bs1_tm60.json',
        'Small Model-3': 'log_pssr_nss_FFFT_bs3_tm60.json',
        'Small Model-5': 'log_pssr_nss_FFFT_bs5_tm60.json',

        'Standard Model-1': 'log_pssr_nss_FFFF_bs1_tm60.json',
        'Standard Model-3': 'log_pssr_nss_FFFF_bs3_tm60.json',
        'Standard Model-5': 'log_pssr_nss_FFFF_bs5_tm60.json'
    }

    problem_level = {}  # map problem_id to level
    level_map = {}  # map t_length to level (start from 0)
    for i in range(level):
        for j in range(span):
            level_map[i * span + j + 1] = i + 1
    for pid in range(1, config['data']['max_pid'] + 1):
        t_length = len(load_json(f'../../datasets/problems/{pid}.json')['theorem_seqs'])
        problem_level[pid] = level_map[t_length] if t_length <= level * span else level

    method_name_max_len = max([len(m) for m in filenames.keys()])
    # correct_problem_ids = load_json('../../outputs/log/log_data_generate_synthetic_data.json')['generated']
    print("------------------------------------------------------------------------")
    for method in filenames.keys():  # pssr_log
        outputs = [method + "".join([" "] * (method_name_max_len - len(method)))]

        try:
            pssr_log = load_json(f"../../outputs/log/{filenames[method]}")

            total_level_count = [0 for _ in range(level + 1)]  # [total, l1, l2, ...]
            solved_level_count = [0 for _ in range(level + 1)]
            for pid in pssr_log["total"]:
                # if str(pid) not in correct_problem_ids:  # skip problem with wrong annotations
                #     continue
                total_level_count[0] += 1
                total_level_count[problem_level[pid]] += 1
                if str(pid) in pssr_log["solved"]:
                    solved_level_count[0] += 1
                    solved_level_count[problem_level[pid]] += 1

            for i in range(level + 1):
                if total_level_count[i] == 0:
                    outputs.append('Nan')
                else:
                    outputs.append(str(round(solved_level_count[i] / total_level_count[i] * 100, 2)))

                outputs[-1] = outputs[-1] + ' ' * (5 - len(outputs[-1]))

            if latex:
                print(' & '.join(outputs))
            else:
                print(' | '.join(outputs))

            if method in ["BART-base", "DualGeoSolver", "FGeo-HyperGNet", "Backward-RS", 'Kimi-K2', 'Ours',
                          'Text-only-5', 'Forward-only-5', 'No Gate-5', 'Small Model-5']:
                print("------------------------------------------------------------------------")
        except Exception:
            pass
    print("------------------------------------------------------------------------")


def get_args():
    parser = argparse.ArgumentParser(description="Welcome to use GPS!")
    parser.add_argument("--func", type=str, required=True,
                        choices=["get_acc", "kill"], help="function that you want to run")
    parser.add_argument("--latex", action="store_true", default=False)
    parser.add_argument("--filename", type=str, required=False,
                        help="python filename that you want to kill")

    parsed_args = parser.parse_args()
    print(f"args: {str(parsed_args)}\n")
    return parsed_args


if __name__ == '__main__':
    """
    python tools.py --func get_acc --latex
    """
    args = get_args()
    if args.func == "get_acc":
        get_acc(latex=args.latex)
    elif args.func == "kill":
        kill(args.filename)
