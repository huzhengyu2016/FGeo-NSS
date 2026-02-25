from itertools import combinations
from tools import replace_paras, parse_fact, replace_expr, _satisfy_algebraic
from tools import _anti_parse_operation, _serialize_operation, _serialize_fact
from sympy import symbols, nonlinsolve, FiniteSet, EmptySet
from func_timeout import func_timeout, FunctionTimedOut


class Problem:
    def __init__(self, parsed_gdl, parsed_cdl, timeout=5):
        self.parsed_gdl = parsed_gdl
        self.parsed_cdl = parsed_cdl
        self.timeout = timeout

        # forward related
        self.facts = []  # fact_id -> (predicate, instance, {premise_id}, operation_id)
        self.fact_id = {}  # (predicate, instance) -> fact_id
        self.predicate_to_fact_instances = {}  # predicate -> [fact_instance]

        # backward related
        self.goals = []  # goal_id -> (predicate, instance, father_id, operation_id)
        self.status_of_goal = []  # goal_id -> int (0: not check, 1: solved, -1: skip or unsolved)
        self.goal_ids = {}  # (predicate, instance) -> {goal_id}
        self.premise_ids_of_goal = {}  # sub_goal_id -> {premise_id}
        self.sub_operations = {}  # goal_id -> {sub_goal_operation_id}
        self.predicate_to_goal_instances = {}  # predicate -> [goal_instance]

        # forward and backward
        self.operations = []  # operation_id -> (operation_type, operation_predicate, operation_instance)
        self.operation_groups = []  # operation_id -> {fact_id} or {goal_id}
        self.theorem_instances = {}  # theorem_name -> {(theorem_paras, premises, conclusion)}

        # algebraic system
        self.points = {}  # point_sym -> point_value
        self.sym_to_value = {}  # sym -> value
        self.sym_to_sym = {}  # multiple_sym -> unified_sym
        self.sym_to_syms = {}  # unified_sym -> {multiple_sym}
        self.equations = {}  # group_id -> ((simplified_eq), ({premise_id}), {sym})
        self.group_count = 0  # generate group_id
        self.simplified_algebraic_goal = {}  # goal_id -> (simplified_eq, {premise_id}, {dependent_sym}, {group_id})
        self.solved_target_cache = {}  # target_expr -> (status, {premise_id})
        self.attempted_equations_cache = set()  # {(target_dependent_equation)}

        # init problem
        self._construct()

    def _construct(self):
        # 1. init problem
        for predicate in list(self.parsed_gdl['Presets']) + list(self.parsed_gdl['Relations']):
            if predicate == 'Eq':
                continue
            self.predicate_to_fact_instances[predicate] = []
            self.predicate_to_goal_instances[predicate] = []
        self.predicate_to_fact_instances['Eq'] = []
        self.predicate_to_goal_instances['Eq'] = []

        # 2. add construction cdl
        premise_ids = set()
        operation_id = self._add_operation(('Preset', 'init_construction', None))
        for predicate, instance in self.parsed_cdl['construction_cdl']:
            fact_id, _ = self._add_fact(predicate, instance, (), operation_id)
            if fact_id is not None:
                premise_ids.add(fact_id)

        # 3. add point's coordinate
        for point in self.parsed_cdl['points']:
            self.points[symbols(f'{point}.x')] = self.parsed_cdl['points'][point][0]
            self.points[symbols(f'{point}.y')] = self.parsed_cdl['points'][point][1]

        # 4. topological extend
        shapes = set()
        collinears = set()
        collinears_raw = set()
        extend_constructions = {
            'Point': set(),
            'Line': set(),
            'PointOnLine': set(),
            'Angle': set(),
            'Triangle': set(),
            'Quadrilateral': set(),
            'Circle': set(),
            'PointOnCircle': set(),
            'DoublePointsOnCircle': set(),
            'TriplePointsOnCircle': set(),
            'QuadruplePointsOnCircle': set()
        }

        # 4.1 Collinear extend
        for instance in self.predicate_to_fact_instances['Collinear']:
            for point in instance:  # add points
                extend_constructions['Point'].add((point,))

            collinears_raw.add(instance)
            collinears_raw.add(instance[::-1])
            for a, b in combinations(instance, 2):
                extend_constructions['Line'].add((a, b))
                extend_constructions['Line'].add((b, a))
            for a, b, c in combinations(instance, 3):
                extend_constructions['PointOnLine'].add((b, a, c))
                extend_constructions['PointOnLine'].add((b, c, a))
                collinears.add((a, b, c))
                collinears.add((c, b, a))

        # 4.2 Cocircular extend
        for instance in self.predicate_to_fact_instances['Cocircular']:
            extend_constructions['Circle'].add((instance[0],))
            circle = instance[0]
            points = instance[1:]
            for point in points:
                extend_constructions['Point'].add((point,))
                extend_constructions['PointOnCircle'].add((point, circle))
            if len(points) >= 2:
                for a, b in combinations(points, 2):
                    extend_constructions['DoublePointsOnCircle'].add((a, b, circle))
                    extend_constructions['DoublePointsOnCircle'].add((b, a, circle))
            if len(points) >= 3:
                for a, b, c in combinations(points, 3):
                    extend_constructions['TriplePointsOnCircle'].add((a, b, c, circle))
                    extend_constructions['TriplePointsOnCircle'].add((b, c, a, circle))
                    extend_constructions['TriplePointsOnCircle'].add((c, a, b, circle))
            if len(points) >= 4:
                for a, b, c, d in combinations(points, 4):
                    extend_constructions['QuadruplePointsOnCircle'].add((a, b, c, d, circle))
                    extend_constructions['QuadruplePointsOnCircle'].add((b, c, d, a, circle))
                    extend_constructions['QuadruplePointsOnCircle'].add((c, d, a, b, circle))
                    extend_constructions['QuadruplePointsOnCircle'].add((d, a, b, c, circle))

        # 4.3 Shape extend (combination)
        jigsaw_unit = {}  # shape's jigsaw
        shape_unit = []  # mini shape unit
        for instance in self.predicate_to_fact_instances['Shape']:  # Shape
            for i in range(len(instance)):  # add point, line, and angles
                if len(instance[i]) == 1:  # point
                    extend_constructions['Point'].add((instance[i],))
                elif len(instance[i]) == 2:  # line
                    extend_constructions['Point'].add((instance[i][0],))
                    extend_constructions['Point'].add((instance[i][1],))
                    extend_constructions['Line'].add(tuple(instance[i]))
                    extend_constructions['Line'].add(tuple(instance[i][::-1]))
                    j = (i + 1) % len(instance)  # add init angle
                    if len(instance[j]) == 2:
                        extend_constructions['Angle'].add((instance[i][0], instance[i][1], instance[j][1]))
                else:  # arc
                    extend_constructions['Point'].add((instance[i][1],))
                    extend_constructions['Point'].add((instance[i][2],))

            multiple_forms = {instance}
            for bias in range(1, len(instance)):  # all forms
                multiple_form = tuple([instance[(i + bias) % len(instance)] for i in range(len(instance))])
                multiple_forms.add(multiple_form)

            shapes.update(multiple_forms)
            for shape in multiple_forms:
                jigsaw_unit[shape] = multiple_forms
                shape_unit.append(shape)
        shape_comb = shape_unit
        jigsaw_comb = jigsaw_unit
        while len(shape_comb):
            shape_comb_new = []
            jigsaw_comb_new = {}
            for unit in shape_unit:
                for comb in shape_comb:
                    if len(unit) == 0 or len(comb) == 0:
                        continue
                    if len(unit[-1]) != len(comb[0]):  # has same sides?
                        continue
                    elif len(unit[-1]) == 3:  # is arc and same?
                        if unit[-1] != comb[0]:
                            continue
                    else:
                        if unit[-1] != comb[0][::-1]:  # is line and same?
                            continue

                    if unit in jigsaw_comb[comb]:  # comb is combined from unit
                        continue

                    same_length = 1  # number of same sides
                    mini_length = len(unit) if len(unit) < len(comb) else len(comb)  # mini length
                    while same_length < mini_length:
                        if len(unit[- same_length - 1]) != len(comb[same_length]):  # all arcs or all lines
                            break
                        elif len(unit[- same_length - 1]) == 3:  # arc
                            if unit[- same_length - 1] != comb[same_length]:
                                break
                        else:  # line
                            if unit[- same_length - 1] != comb[same_length][::-1]:
                                break

                        same_length += 1

                    new_shape = list(unit[0:len(unit) - same_length])  # diff sides in polygon1
                    new_shape += list(comb[same_length:len(comb)])  # diff sides in polygon2

                    if not len(new_shape) == len(set(new_shape)):  # ensure no ring
                        continue

                    new_shape = tuple(new_shape)
                    if new_shape in shapes:
                        continue

                    all_sides = ""
                    for item in new_shape:  # remove circle center point
                        if len(item) == 3:
                            item = item[1:]
                        all_sides += item
                    checked = True
                    for point in all_sides:
                        if all_sides.count(point) > 2:
                            checked = False
                            break
                    if not checked:  # ensure no holes
                        continue

                    if new_shape in shapes:
                        continue

                    multiple_forms = {new_shape}
                    for bias in range(1, len(new_shape)):  # all forms
                        multiple_form = tuple([new_shape[(i + bias) % len(new_shape)] for i in range(len(new_shape))])
                        multiple_forms.add(multiple_form)
                    shapes.update(multiple_forms)

                    new_shape_jigsaw = jigsaw_unit[unit] | jigsaw_comb[comb]
                    for shape in multiple_forms:
                        jigsaw_comb_new[shape] = new_shape_jigsaw
                        shape_comb_new.append(shape)

            shape_comb = shape_comb_new
            jigsaw_comb = jigsaw_comb_new

        # 4.4 Angle expand (combination)
        angle_unit = list(extend_constructions['Angle'])
        jigsaw_unit = {}
        for angle in angle_unit:
            jigsaw_unit[angle] = {angle}
        angle_comb = angle_unit  # combination angle
        jigsaw_comb = jigsaw_unit  # angle's jigsaw
        while len(angle_comb):
            angle_comb_new = []
            jigsaw_comb_new = {}
            for unit in angle_unit:
                for comb in angle_comb:

                    if unit in jigsaw_comb[comb]:  # comb is combined from unit
                        continue

                    if not (unit[1] == comb[1] and unit[2] == comb[0] and unit[0] != comb[2]):  # ensure adjacent
                        continue

                    if (unit[0], unit[1], comb[2]) in extend_constructions['Angle'] or \
                            (unit[0], comb[2], unit[1]) in extend_constructions['Angle'] or \
                            (comb[2], unit[0], unit[1]) in extend_constructions['Angle']:
                        continue

                    new_angle = (unit[0], unit[1], comb[2])

                    if not len(new_angle) == len(set(new_angle)):  # ensure same points
                        continue

                    if new_angle in extend_constructions['Angle']:
                        continue
                    extend_constructions['Angle'].add(new_angle)

                    new_angle_jigsaw = jigsaw_unit[unit] | jigsaw_comb[comb]
                    jigsaw_comb_new[new_angle] = new_angle_jigsaw
                    angle_comb_new.append(new_angle)

            angle_comb = angle_comb_new
            jigsaw_comb = jigsaw_comb_new

        # 4.5 add angle, triangle, and quadrilateral
        for shape in shapes:
            # print(shape)
            shape = list(shape)

            for i in range(len(shape)):  # add angles
                j = (i + 1) % len(shape)
                if not (len(shape[i]) == 2 and len(shape[j]) == 2 and shape[i][1] == shape[j][0]):
                    continue
                extend_constructions['Angle'].add((shape[i][0], shape[i][1], shape[j][1]))

            i = 0
            has_arc = False
            while i < len(shape):
                if len(shape[i]) != 2:
                    has_arc = True
                    break
                j = (i + 1) % len(shape)
                if (shape[i][0], shape[i][1], shape[j][1]) in collinears:
                    shape[i] = shape[i][0] + shape[j][1]
                    shape.pop(j)
                    continue
                i += 1

            if has_arc or len(shape) not in {3, 4}:  # only care about triangle and quadrilateral
                continue

            valid = True
            i = 0
            while i < len(shape):
                if shape[i][1] != shape[(i + 1) % len(shape)][0]:
                    valid = False
                    break
                i += 1

            if not valid:
                continue

            polygon = tuple([item[0] for item in shape])
            if len(polygon) == 3:
                extend_constructions['Triangle'].add(polygon)
            else:
                extend_constructions['Quadrilateral'].add(polygon)

        # 4.6 Angle expand (ABC -> CBA)
        for angle in list(extend_constructions['Angle']):
            extend_constructions['Angle'].add((angle[2], angle[1], angle[0]))

        # 4.7 Angle from collinear extend
        for instance in self.predicate_to_fact_instances['Collinear']:
            for a, b, c in combinations(instance, 3):
                extend_constructions['Angle'].add((a, b, c))
                extend_constructions['Angle'].add((c, b, a))

        # 4.8 Angle collinear expand (set same angle to same sym)
        for angle in list(extend_constructions['Angle']):
            if angle == ('D', 'B', 'G'):
                pass
            if symbols(''.join(angle) + '.ma') in self.sym_to_sym:
                continue
            a, v, b = angle
            a_points = {a}  # Points collinear with a and on the same side with a
            b_points = {b}
            for collinear in collinears_raw:
                if v not in collinear:
                    continue
                if a in collinear:
                    if collinear.index(v) < collinear.index(a):  # .....V...A..
                        i = collinear.index(v) + 1
                        while i < len(collinear):
                            a_points.add(collinear[i])
                            i += 1
                    else:  # ...A.....V...
                        i = 0
                        while i < collinear.index(v):
                            a_points.add(collinear[i])
                            i += 1
                if b in collinear:
                    if collinear.index(v) < collinear.index(b):  # .....V...B..
                        i = collinear.index(v) + 1
                        while i < len(collinear):
                            b_points.add(collinear[i])
                            i += 1
                    else:  # ...B.....V...
                        i = 0
                        while i < collinear.index(v):
                            b_points.add(collinear[i])
                            i += 1

            sym = symbols(''.join(angle) + '.ma')
            self.sym_to_syms[sym] = {sym}
            for a_point in a_points:
                for b_point in b_points:
                    angle = (a_point, v, b_point)
                    extend_constructions['Angle'].add(angle)
                    multiple_sym = symbols(''.join(angle) + f'.ma')
                    self.sym_to_sym[multiple_sym] = sym
                    self.sym_to_syms[sym].add(multiple_sym)

        # 4.9 add extended constructions
        operation_id = self._add_operation(('Preset', 'extend_construction', None))
        for predicate in extend_constructions:
            for instance in extend_constructions[predicate]:
                self._add_fact(predicate, instance, premise_ids, operation_id)

        # 5.Add facts
        operation_id = self._add_operation(('Preset', 'init_fact', None))
        for predicate, instance in self.parsed_cdl['relation_cdl']:
            if not self._pass_geometric_constraints(predicate, instance):
                raise Exception(f'EE check not passed when add init fact {(predicate, instance)}.')
            if (predicate, instance) in self.fact_id:
                continue
            fact_id, _ = self._add_fact(predicate, instance, (), operation_id)
            if fact_id is None:
                raise Exception(f'Error when add init fact {(predicate, instance)}.')

        # 6.Set goal
        init_goal_operation_id = self._add_operation(('Preset', 'init_goal', None))
        goal_ids = self._add_goals([self.parsed_cdl['goal_cdl']], None, init_goal_operation_id)
        if goal_ids is None:
            raise Exception(f"Error when set init goal {self.parsed_cdl['goal_cdl']}.")
        self._check_goals(goal_ids)

    def _add_fact(self, predicate, instance, premise_ids, operation_id):
        if predicate == 'Eq':
            instance = self._adjust_expr(instance)
            if instance is None or len(instance.free_symbols) == 0:
                return None, set()

        if (predicate, instance) in self.fact_id:
            return None, set()

        fact_id = len(self.facts)
        self.facts.append((predicate, instance, set(premise_ids), operation_id))
        self.fact_id[(predicate, instance)] = fact_id
        self.predicate_to_fact_instances[predicate].append(instance)
        self.operation_groups[operation_id].add(fact_id)

        # # theorem auto apply
        # if predicate in self.parsed_gdl['Relations']:
        #     predicate_lower = predicate[0].lower() + predicate[1:]
        #     for i in range(len(predicate_lower)):
        #         if predicate_lower[i].isupper():
        #             predicate_lower = predicate_lower.replace(predicate_lower[i], '_' + predicate_lower[i].lower())
        #
        #     theorem_multiple_form_keyword = predicate_lower + '_multiple_forms'
        #     theorem_auto_extend_keyword = predicate_lower + '_auto_extend'
        #     theorem_multiple_forms = []
        #     theorem_auto_extend = []
        #     for theorem_name in self.parsed_gdl['Theorems']:
        #         if theorem_name.startswith(theorem_multiple_form_keyword):
        #             theorem_multiple_forms.append(theorem_name + '(' + ','.join(instance) + ')')
        #         elif theorem_name.startswith(theorem_auto_extend_keyword):
        #             theorem_auto_extend.append(theorem_name + '(' + ','.join(instance) + ')')
        #
        #     for theorem in theorem_multiple_forms + theorem_auto_extend:
        #         self.apply(theorem)

        if predicate != 'Eq':
            goal_ids = set()
            if (predicate, instance) in self.goal_ids:
                goal_ids = self.goal_ids[(predicate, instance)]
            return fact_id, goal_ids

        if self.operations[operation_id] == ('Preset', 'solve_eq', None):
            return fact_id, set()

        new_simplified_eqs = [instance]
        new_premise_ids_list = [{fact_id}]
        new_syms = set()

        # replace solved sym with its value
        for sym in instance.free_symbols:
            if sym in self.sym_to_value:
                new_simplified_eqs[0] = new_simplified_eqs[0].subs(sym, self.sym_to_value[sym])
                new_premise_ids_list[0].add(self.fact_id[('Eq', sym - self.sym_to_value[sym])])
            else:
                new_syms.add(sym)

        if len(new_syms) == 0:  # no unsolved sym
            return fact_id, set()

        # print(new_syms)

        # merge equations group
        deleted_group_ids = set()
        for group_id in self.equations:
            simplified_eqs, premise_ids_list, syms = self.equations[group_id]
            if len(new_syms & syms) > 0:
                deleted_group_ids.add(group_id)
                new_simplified_eqs.extend(simplified_eqs)
                new_premise_ids_list.extend(premise_ids_list)
                new_syms.update(syms)

        # print("self.equations:", self.equations)
        # print("instance:", instance)
        # print("deleted_group_ids:", deleted_group_ids)
        for group_id in deleted_group_ids:  # delete old groups
            del self.equations[group_id]

        goal_ids = set()  # influenced sub_goals
        for goal_id in self.simplified_algebraic_goal:
            if len(new_syms & self.simplified_algebraic_goal[goal_id][2]) > 0:
                goal_ids.add(goal_id)
        # print("goal_ids:", goal_ids)
        # print()
        # solve equations
        new_syms = sorted(list(new_syms), key=str)
        new_simplified_eqs = sorted(new_simplified_eqs, key=str)
        solved_values = {}

        try:
            solutions = func_timeout(timeout=self.timeout, func=nonlinsolve, args=(new_simplified_eqs, new_syms))
            # print(new_simplified_eqs)
            # print(new_syms)
            # print(solutions)
            # print()

            # print(solutions)
            if solutions is not EmptySet and type(solutions) is FiniteSet and len(solutions) > 0:
                solutions = list(solutions)
                for i in range(len(solutions))[::-1]:  # remove the negative solutions
                    for j in range(len(new_syms)):
                        if len(solutions[i][j].free_symbols) > 0:  # skip unsolved sym
                            continue
                        if '.' not in str(new_syms[j]):  # skip free symbols
                            continue
                        try:
                            if _satisfy_algebraic['L'](solutions[i][j]):
                                solutions.pop(i)
                                break
                        except BaseException as e:
                            pass

                for j in range(len(new_syms)):
                    if len(solutions[0][j].free_symbols) != 0:  # no numeric solution
                        continue

                    same = True
                    for i in range(1, len(solutions)):
                        if not _satisfy_algebraic['Eq'](solutions[i][j] - solutions[0][j]):
                            same = False
                            break
                    if not same:  # numeric solution not same in every solved result
                        continue

                    try:
                        float(solutions[0][j])
                    except BaseException:
                        pass
                    else:
                        solved_values[new_syms[j]] = solutions[0][j]  # save solved value

            # print(solutions)
            # print()
        except BaseException:
            pass

        # split equations group
        if len(solved_values) == 0:  # no solved value
            self.equations[self.group_count] = (tuple(new_simplified_eqs), tuple(new_premise_ids_list), set(new_syms))
            self.group_count += 1
            return fact_id, goal_ids

        operation_id = self._add_operation(('Preset', 'solve_eq', None))  # add the solved values
        premise_ids = set()
        for new_premise_ids in new_premise_ids_list:
            premise_ids.update(new_premise_ids)
        for sym in solved_values:
            instance = sym - solved_values[sym]
            self.sym_to_value[sym] = solved_values[sym]
            self._add_fact('Eq', instance, premise_ids, operation_id)

        for i in range(len(new_simplified_eqs)):  # replace sym with it's solved value
            for sym in new_simplified_eqs[i].free_symbols:
                if sym in solved_values:
                    new_simplified_eqs[i] = new_simplified_eqs[i].subs(sym, solved_values[sym])
                    new_premise_ids_list[i].add(self.fact_id[('Eq', sym - solved_values[sym])])

        while len(new_simplified_eqs) > 0:
            if len(new_simplified_eqs[0].free_symbols) == 0:  # no unsolved sym, skip
                new_simplified_eqs.pop(0)
                new_premise_ids_list.pop(0)
                continue

            simplified_eqs = [new_simplified_eqs.pop(0)]
            premise_ids_list = [new_premise_ids_list.pop(0)]
            syms = set(simplified_eqs[0].free_symbols)
            update = True
            while update:
                update = False
                for i in range(len(new_simplified_eqs))[::-1]:
                    if len(syms & new_simplified_eqs[i].free_symbols) > 0:
                        simplified_eqs.append(new_simplified_eqs.pop(i))
                        premise_ids_list.append(new_premise_ids_list.pop(i))
                        syms.update(simplified_eqs[-1].free_symbols)
                        update = True
            self.equations[self.group_count] = (tuple(simplified_eqs), tuple(premise_ids_list), set(syms))
            self.group_count += 1

        return fact_id, goal_ids

    def _add_operation(self, operation):
        operation_id = len(self.operations)
        self.operations.append(operation)
        self.operation_groups.append(set())
        return operation_id

    def _adjust_expr(self, expr):
        """添加代数型的fact或goal时，都要先调整expr，替换统一的符号表示，并调整为首项不为负号."""
        for sym in list(expr.free_symbols):
            if sym in self.sym_to_sym:
                replace_sym = self.sym_to_sym[sym]
                if sym != replace_sym:
                    expr = expr.subs(sym, replace_sym)
            elif '.' not in str(sym):  # free symbols
                self.sym_to_sym[sym] = sym
                self.sym_to_syms[sym] = {sym}
            else:
                entities, attr = str(sym).split('.')
                replace = dict(zip(self.parsed_gdl['Attributions'][attr]['paras'], entities))

                for predicate, paras in self.parsed_gdl['Attributions'][attr]['geometric_constraints']:
                    if (predicate, replace_paras(paras, replace)) not in self.fact_id:
                        # print(expr)
                        # print(self.predicate_to_fact_instances[predicate])
                        # print((predicate, replace_paras(paras, replace)))
                        # print()
                        return None

                self.sym_to_sym[sym] = sym
                self.sym_to_syms[sym] = {sym}
                for paras in self.parsed_gdl['Attributions'][attr]['multiple_forms']:
                    multiple_sym = symbols(''.join(replace_paras(paras, replace)) + '.' + attr)
                    self.sym_to_sym[multiple_sym] = sym
                    self.sym_to_syms[sym].add(multiple_sym)

        if expr != 0 and str(expr)[0] == '-':
            expr = -expr

        return expr

    def _run_gpl(self, theorem_gpl):
        paras = []
        instances = [[]]
        premise_ids = [[]]

        for gpl_one_term in theorem_gpl:
            product = gpl_one_term['product']  # (predicate, paras, inherent_same_index, mutual_same_index, added_index)
            geometric_premises = gpl_one_term['geometric_premises']  # [(predicate, paras)]
            algebraic_premises = gpl_one_term['algebraic_premises']  # [expr]
            algebraic_constraints = gpl_one_term['algebraic_constraints']  # [(relation_type, expr)]

            new_instances = []
            new_premise_ids = []
            paras.extend([product[1][j] for j in product[4]])
            for k in range(len(instances)):
                instance = instances[k]
                for product_instance in self.predicate_to_fact_instances[product[0]]:
                    # check inherent same index constraint
                    passed = True
                    for i, j in product[2]:
                        if product_instance[i] != product_instance[j]:
                            passed = False
                            break
                    if not passed:
                        continue

                    # check mutual same index constraint
                    passed = True
                    for i, j in product[3]:
                        if instance[i] != product_instance[j]:
                            passed = False
                            break
                    if not passed:
                        continue

                    # constrained cartesian product: add different letter
                    new_instance = list(instance)
                    new_instance.extend([product_instance[j] for j in product[4]])

                    replace = dict(zip(paras, new_instance))

                    # check constraints
                    passed, constraints_premise_id = self._pass_constraints(
                        geometric_premises, algebraic_premises, algebraic_constraints, replace)
                    if not passed:
                        continue

                    new_premise_id = list(premise_ids[k])
                    new_premise_id.append(self.fact_id[(product[0], product_instance)])
                    new_premise_id.extend(constraints_premise_id)

                    new_instances.append(new_instance)
                    new_premise_ids.append(new_premise_id)

            instances = new_instances
            premise_ids = new_premise_ids

        return paras, instances, premise_ids

    def _pass_constraints(self, geometric_premises, algebraic_premises, algebraic_constraints, replace):
        premise_ids = set()

        # check geometric premises
        for predicate, paras in geometric_premises:
            fact = (predicate, replace_paras(paras, replace))
            # print(fact)
            # print(fact in self.fact_id)
            # print()
            if fact not in self.fact_id:
                return False, None
            premise_ids.add(self.fact_id[fact])

        # check algebraic constraint of dependent entity
        for algebraic_relation, expr in algebraic_constraints:
            expr = replace_expr(expr, replace)
            if not _satisfy_algebraic[algebraic_relation](expr, self.points):
                return False, None

        # check algebraic premises
        for expr in algebraic_premises:
            expr = replace_expr(expr, replace)

            status, algebraic_premise_ids = self._pass_algebraic_premise(expr)
            if status != 1:
                return False, None

            premise_ids.update(algebraic_premise_ids)

        return True, premise_ids

    def _pass_algebraic_premise(self, expr):
        """return status, premise_ids
        status=1 solved
        status=-1 unsolved
        status=0 no solution
        """
        expr = self._adjust_expr(expr)
        if expr is None:
            return -1, set()

        if ('Eq', expr) in self.fact_id:  # expr in self.facts
            return 1, {self.fact_id[('Eq', expr)]}

        premise_ids = set()
        for sym in list(expr.free_symbols):
            if sym in self.sym_to_value:
                premise_ids.add(self.fact_id[('Eq', sym - self.sym_to_value[sym])])
                expr = expr.subs(sym, self.sym_to_value[sym])

        if len(expr.free_symbols) == 0:
            if _satisfy_algebraic['Eq'](expr):
                return 1, premise_ids
            return -1, premise_ids

        if expr in self.solved_target_cache:
            status, cache_premise_ids = self.solved_target_cache[expr]
            return status, cache_premise_ids | premise_ids

        target_sym = symbols('t')
        equations = [target_sym - expr]
        syms = set(expr.free_symbols)
        for group_id in self.equations:
            if len(self.equations[group_id][2] & expr.free_symbols) > 0:
                equations.extend(self.equations[group_id][0])
                for eq_premise_ids in self.equations[group_id][1]:
                    premise_ids.update(eq_premise_ids)
                syms.update(self.equations[group_id][2])
        syms = [target_sym] + sorted(list(syms), key=str)
        equations = sorted(equations, key=str)

        equations_tuple = tuple(equations)
        if equations_tuple in self.attempted_equations_cache:
            return 0, None
        self.attempted_equations_cache.add(equations_tuple)

        try:
            equation_solutions = func_timeout(
                timeout=self.timeout,
                func=nonlinsolve,
                args=(equations, syms)
            )
            # print(expr)
            # print(equations)
            # print(equation_solutions)
            # print()
        except FunctionTimedOut:
            return 0, None
        except Exception:
            return 0, None

        # print(equation_solutions is EmptySet)
        # print(type(equation_solutions) is not FiniteSet)
        if equation_solutions is EmptySet or type(equation_solutions) is not FiniteSet:
            return 0, None

        equation_solutions = list(equation_solutions)
        for i in range(len(equation_solutions))[::-1]:  # remove the negative solutions
            for j in range(len(syms)):
                if len(equation_solutions[i][j].free_symbols) > 0:  # skip unsolved sym
                    continue
                if '.' not in str(syms[j]):  # skip free symbols
                    continue
                try:
                    if _satisfy_algebraic['L'](equation_solutions[i][j]):
                        equation_solutions.pop(i)
                        break
                except Exception as e:
                    pass

        for solved_value in equation_solutions:  # in every solution, the solved value of target_sym must be 0
            if len(solved_value[0].free_symbols) > 0:
                return 0, None
            if not _satisfy_algebraic['Eq'](solved_value[0]):
                self.solved_target_cache[expr] = (-1, premise_ids)
                return -1, premise_ids

        self.solved_target_cache[expr] = (1, premise_ids)
        return 1, premise_ids

    def _add_conclusion(self, theorem_gdl, replace, premise_ids, operation_id):
        # print(replace)
        predicate, instance = theorem_gdl['conclusion']
        if predicate == "Eq":
            instance = replace_expr(instance, replace)
        else:
            instance = replace_paras(instance, replace)
        return self._add_fact(predicate, instance, premise_ids, operation_id)

    def _add_goals(self, goals, father_id, operation_id):
        for predicate, instance in goals:
            if predicate == 'Eq':
                # print(instance)
                instance = self._adjust_expr(instance)
                # print(instance)
                # print()
                if instance is None:
                    return None
            if self._ancestor_has_goal(predicate, instance, father_id):  # ensure ancestor no sub_goal
                return None
            if father_id is not None:
                for father_sub_operation_id in self.sub_operations[father_id]:  # ensure father no same sub_goal
                    if self.operations[father_sub_operation_id] == self.operations[operation_id]:
                        return None

        goal_ids = []
        for predicate, instance in goals:  # add goal
            if predicate == 'Eq':
                instance = self._adjust_expr(instance)
            goal_id = len(self.goals)
            self.goals.append((predicate, instance, father_id, operation_id))
            # print(goal_id, (predicate, instance), len(self.goals))
            self.status_of_goal.append(0)
            if (predicate, instance) not in self.goal_ids:
                self.goal_ids[(predicate, instance)] = {goal_id}
            else:
                self.goal_ids[(predicate, instance)].add(goal_id)
            self.sub_operations[goal_id] = set()
            if father_id is not None:
                self.sub_operations[father_id].add(operation_id)
            self.predicate_to_goal_instances[predicate].append(instance)
            self.operation_groups[operation_id].add(goal_id)
            if predicate == 'Eq':
                self.simplified_algebraic_goal[goal_id] = (instance, set(), set(), set())
            goal_ids.append(goal_id)

        # print(goal_ids)
        return goal_ids

    def _ancestor_has_goal(self, check_predicate, check_instance, goal_id):
        if goal_id is None:
            return False

        predicate, instance, father_id, _ = self.goals[goal_id]
        if predicate == check_predicate and instance == check_instance:
            return True

        return self._ancestor_has_goal(check_predicate, check_instance, father_id)

    def _pass_geometric_constraints(self, predicate, instance):
        if predicate == 'Eq':  # eq
            for sym in instance.free_symbols:
                sym = str(sym)
                if '.' not in sym:
                    continue
                instance, attr = sym.split('.')
                attr_gdl = self.parsed_gdl["Attributions"][attr]
                replace = dict(zip(attr_gdl['paras'], instance))
                for ee_check_predicate, ee_check_paras in attr_gdl['geometric_constraints']:
                    ee_check_instance = tuple(replace_paras(ee_check_paras, replace))
                    if (ee_check_predicate, ee_check_instance) not in self.fact_id:
                        # print(predicate, instance)
                        # print(self.predicate_to_fact_instances[ee_check_predicate])
                        # print((ee_check_predicate, ee_check_instance))
                        # print()
                        return False
        elif predicate in self.parsed_gdl["Presets"]:  # Presets
            return (predicate, instance) in self.fact_id
        else:  # relation

            relation_gdl = self.parsed_gdl["Relations"][predicate]
            replace = dict(zip(relation_gdl['paras'], instance))
            for ee_check_predicate, ee_check_paras in relation_gdl['geometric_constraints']:
                ee_check_instance = replace_paras(ee_check_paras, replace)
                if (ee_check_predicate, ee_check_instance) not in self.fact_id:
                    # print(predicate, instance)
                    # print(self.predicate_to_fact_instances[ee_check_predicate])
                    # print((ee_check_predicate, ee_check_instance))
                    # print()
                    return False
        return True

    def _pass_algebraic_constraints(self, theorem_gdl, replace):
        for gpl_one_term in theorem_gdl['premises_gpl']:
            for algebraic_relation, expr in gpl_one_term['algebraic_constraints']:
                expr = replace_expr(expr, replace)
                if not _satisfy_algebraic[algebraic_relation](expr, self.points):
                    return False
        return True

    def _find_father_ids(self, predicate, instance):
        # print('goal:', predicate, instance)
        father_ids = []
        if predicate != 'Eq':
            if (predicate, instance) not in self.goal_ids:
                return []
            for goal_id in self.goal_ids[(predicate, instance)]:
                if self.status_of_goal[goal_id] == 0:
                    father_ids.append(goal_id)
        else:  # algebraic goal
            for goal_id in self.simplified_algebraic_goal:
                # print(self.simplified_algebraic_goal[goal_id])
                if self.status_of_goal[goal_id] != 0:
                    continue
                if len(self.simplified_algebraic_goal[goal_id][2] & instance.free_symbols) == 0:
                    continue
                father_ids.append(goal_id)

        return father_ids

    def _generate_sub_goals(self, theorem_gdl, replace):
        sub_goals = []

        for gpl_one_term in theorem_gdl['premises_gpl']:
            predicate, paras = gpl_one_term['product'][:2]
            instance = tuple(replace_paras(paras, replace))
            if not self._pass_geometric_constraints(predicate, instance):
                return None
            sub_goals.append((predicate, instance))

            for predicate, paras in gpl_one_term['geometric_premises']:
                instance = tuple(replace_paras(paras, replace))
                if not self._pass_geometric_constraints(predicate, instance):
                    return None
                sub_goals.append((predicate, instance))

            for expr in gpl_one_term['algebraic_premises']:
                instance = replace_expr(expr, replace)
                instance = self._adjust_expr(instance)
                if instance is None:  # not ask len(instance.free_symbols) > 0
                    return None
                if not self._pass_geometric_constraints('Eq', instance):
                    return None
                sub_goals.append(('Eq', instance))

        return sub_goals

    def _generate_theorem_instances(self, theorem_name):
        """only for algebraic conclusion"""
        theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]
        paras, instances, _ = self._run_gpl(theorem_gdl['entities_gpl'])
        # print(theorem_name, instances)
        # print()

        self.theorem_instances[theorem_name] = set()
        for i in range(len(instances)):
            replace = dict(zip(paras, instances[i]))
            # print(replace)
            # print(theorem_name + '(' + ','.join(replace_paras(theorem_gdl['paras'], replace)) + ')')

            predicate, instance = theorem_gdl['conclusion']  # generate conclusions
            instance = self._adjust_expr(replace_expr(instance, replace))
            if instance is None or len(instance.free_symbols) == 0:
                continue

            sub_goals = self._generate_sub_goals(theorem_gdl, replace)  # generate sub_goals
            if sub_goals is None:
                continue

            theorem = theorem_name + '(' + ','.join(replace_paras(theorem_gdl['paras'], replace)) + ')'

            self.theorem_instances[theorem_name].add((theorem, tuple(sub_goals), instance))

        # print()

    def _set_status(self, goal_id, status):
        """
        1: 向下传递给所有children-1，同时向上检查一层，如果所有兄弟都是 1，就应用定理
        -1:横向传递给所有兄弟、向下传递给所有children
        """
        all_sub_goal_solved = False
        if self.status_of_goal[goal_id] == 0:
            _, _, father_id, operation_id = self.goals[goal_id]

            if status == 1:
                self.status_of_goal[goal_id] = 1
                all_sub_goal_solved = True
                for bro_goal_id in self.operation_groups[operation_id]:
                    if self.status_of_goal[bro_goal_id] != 1:
                        all_sub_goal_solved = False
                        break

            else:  # status == -1
                self.status_of_goal[goal_id] = -1
                for bro_goal_id in self.operation_groups[operation_id]:
                    self._set_status(bro_goal_id, -1)

            if goal_id in self.sub_operations:
                for child_operation_id in self.sub_operations[goal_id]:
                    for child_goal_id in self.operation_groups[child_operation_id]:
                        self._set_status(child_goal_id, -1)

        return all_sub_goal_solved

    def _check_goals(self, goal_ids):
        if len(goal_ids) == 0:
            return

        goal_ids = list(goal_ids)
        for goal_id in goal_ids:
            if self.status_of_goal[goal_id] != 0:
                return
            all_sub_goal_solved = False
            predicate, instance, father_id, operation_id = self.goals[goal_id]

            if self.goals[goal_id][0] == 'Eq':  # algebraic goal
                instance = self.simplified_algebraic_goal[goal_id][0]
                status, premise_ids = self._pass_algebraic_premise(instance)
                if status == 1:  # has solution, update status
                    all_sub_goal_solved = self._set_status(goal_id, 1)
                    premise_ids.update(self.simplified_algebraic_goal[goal_id][1])
                    self.premise_ids_of_goal[goal_id] = premise_ids
                elif status == -1:  # has solution, update status
                    self._set_status(goal_id, -1)
                    premise_ids.update(self.simplified_algebraic_goal[goal_id][1])
                    self.premise_ids_of_goal[goal_id] = premise_ids
                else:  # no solution, simplify expr
                    premise_ids = self.simplified_algebraic_goal[goal_id][1]
                    for sym in list(instance.free_symbols):
                        if sym in self.sym_to_value:
                            instance = instance.subs(self.sym_to_value)
                            premise_ids.add(self.fact_id[('Eq', sym - self.sym_to_value[sym])])
                    dependent_syms = set(instance.free_symbols)
                    group_ids = set()
                    for group_id in self.equations:
                        if len(dependent_syms & self.equations[group_id][2]) > 0:
                            dependent_syms.update(self.equations[group_id][2])
                            group_ids.add(group_id)
                    self.simplified_algebraic_goal[goal_id] = (instance, premise_ids, dependent_syms, group_ids)

            else:  # geometric goal
                # print(f'check={(predicate, instance) in self.fact_id}', (predicate, instance))
                if (predicate, instance) in self.fact_id:
                    self.premise_ids_of_goal[goal_id] = {self.fact_id[(predicate, instance)]}
                    all_sub_goal_solved = self._set_status(goal_id, 1)
                elif predicate in self.parsed_gdl['Presets']:
                    self._set_status(goal_id, -1)

            if all_sub_goal_solved and father_id is not None:
                premise_ids = set()
                for bro_goal_id in self.operation_groups[operation_id]:
                    premise_ids.update(self.premise_ids_of_goal[bro_goal_id])
                _, theorem_name, theorem_paras = self.operations[operation_id]
                operation_id = self._add_operation(('Apply', theorem_name, theorem_paras))
                theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]
                replace = dict(zip(theorem_gdl['paras'], theorem_paras))
                fact_id, new_goal_ids = self._add_conclusion(theorem_gdl, replace, premise_ids, operation_id)
                if fact_id is not None:
                    goal_ids.extend(new_goal_ids)

    def apply(self, theorem):
        if '(' in theorem:
            theorem_name, theorem_paras = parse_fact(theorem)
        else:
            theorem_name = theorem
            theorem_paras = None
        if theorem_name not in self.parsed_gdl["Theorems"]:
            e_msg = f"Unknown theorem name: '{theorem_name}'."
            raise Exception(e_msg)

        update = False
        all_goal_ids = set()

        if theorem_paras is not None:  # parameterized form
            if len(theorem_paras) != len(self.parsed_gdl["Theorems"][theorem_name]['paras']):
                e_msg = f"Theorem '{theorem_name}' has wrong number of paras."
                raise Exception(e_msg)

            theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]
            replace = dict(zip(theorem_gdl['paras'], theorem_paras))
            premise_ids = set()

            for gpl_one_term in theorem_gdl['premises_gpl']:  # run gdl with theorem parameter
                product = gpl_one_term['product']
                algebraic_constraints = gpl_one_term['algebraic_constraints']
                geometric_premises = gpl_one_term['geometric_premises']
                algebraic_premises = gpl_one_term['algebraic_premises']
                predicate = product[0]
                instance = replace_paras(product[1], replace)
                # print((predicate, instance))
                # print((predicate, instance) in self.fact_id)
                # print()
                if (predicate, instance) not in self.fact_id:  # verification mode, not cartesian product
                    return False
                premise_ids.add(self.fact_id[(predicate, instance)])

                # check constraints
                passed, constraints_premise_ids = self._pass_constraints(
                    geometric_premises, algebraic_premises, algebraic_constraints, replace)
                if not passed:
                    return False
                premise_ids.update(constraints_premise_ids)

            # add operation
            operation_id = self._add_operation(('Apply', theorem_name, theorem_paras))

            # add conclusions
            fact_id, goal_ids = self._add_conclusion(theorem_gdl, replace, premise_ids, operation_id)
            if fact_id is not None:
                update = True
                all_goal_ids.update(goal_ids)

        else:  # parameter-free form
            theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]
            paras, instances, premise_ids = self._run_gpl(theorem_gdl['premises_gpl'])
            for i in range(len(instances)):
                replace = dict(zip(paras, instances[i]))

                # add operation
                theorem_paras = replace_paras(theorem_gdl['paras'], replace)
                operation_id = self._add_operation(('Apply', theorem_name, theorem_paras))

                # add conclusions
                fact_id, goal_ids = self._add_conclusion(theorem_gdl, replace, premise_ids[i], operation_id)
                if fact_id is not None:
                    update = True
                    all_goal_ids.update(goal_ids)

        self._check_goals(all_goal_ids)

        return update

    def decompose(self, theorem):
        if '(' in theorem:
            theorem_name, theorem_paras = parse_fact(theorem)
        else:
            theorem_name = theorem
            theorem_paras = None
        if theorem_name not in self.parsed_gdl["Theorems"]:
            e_msg = f"Unknown theorem name: '{theorem_name}'."
            raise Exception(e_msg)

        update = False
        all_goal_ids = set()

        if theorem_paras is not None:  # parameterized form
            if len(theorem_paras) != len(self.parsed_gdl["Theorems"][theorem_name]['paras']):
                e_msg = f"Theorem '{theorem_name}' has wrong number of paras."
                raise Exception(e_msg)

            theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]
            replace = dict(zip(theorem_gdl['paras'], theorem_paras))

            # print(theorem_name, theorem_paras)

            if not self._pass_algebraic_constraints(theorem_gdl, replace):  # ac checks
                return False
            # print('_pass_algebraic_constraints')

            predicate, instance = theorem_gdl['conclusion']  # generate conclusion
            if predicate == "Eq":
                instance = self._adjust_expr(replace_expr(instance, replace))
                if instance is None or len(instance.free_symbols) == 0:
                    return False
            else:
                instance = tuple(replace_paras(instance, replace))

            # print('conclusion:', predicate, instance)

            if not self._pass_geometric_constraints(predicate, instance):  # ee checks
                return False

            # print('_pass_geometric_constraints')

            father_ids = self._find_father_ids(predicate, instance)  # find father nodes
            if len(father_ids) == 0:
                return False

            # print('father_ids:', father_ids)

            sub_goals = self._generate_sub_goals(theorem_gdl, replace)  # generate sub_goals
            if sub_goals is None:
                return False

            for father_id in father_ids:  # add sub_goals
                # print(sub_goals, father_id)
                operation_id = self._add_operation(('Decompose', theorem_name, theorem_paras))
                goal_ids = self._add_goals(sub_goals, father_id, operation_id)
                if goal_ids is not None:
                    update = True
                    all_goal_ids.update(goal_ids)
        else:  # parameter-free form
            theorem_gdl = self.parsed_gdl['Theorems'][theorem_name]
            predicate, paras = theorem_gdl['conclusion']
            # print(theorem_name)
            # print(predicate, paras)
            # print()

            if predicate != 'Eq':  # decompose geometric goal
                # self.show()
                # exit(0)
                theorem_paras = list(paras)
                # print(theorem_name)
                # print()
                same_index = []
                for i in range(len(paras)):
                    for j in range(i + 1, len(paras)):
                        if paras[i] == paras[j]:
                            same_index.append((i, j))
                # print(same_index)

                instance_to_father_ids = {}  # {instance: [father_id]}
                for instance in self.predicate_to_goal_instances[predicate]:
                    for father_id in self.goal_ids[(predicate, instance)]:
                        if self.status_of_goal[father_id] != 0:  # these goal no need to decompose
                            continue

                        pass_check = True  # check instance structure
                        for i, j in same_index:
                            if instance[i] != instance[j]:
                                pass_check = False
                                break
                        if not pass_check:
                            continue

                        if instance in instance_to_father_ids:
                            instance_to_father_ids[instance].add(father_id)
                        else:
                            instance_to_father_ids[instance] = {father_id}

                not_added_paras = set(theorem_gdl['paras']) - set(theorem_paras)
                # print(theorem_gdl['circle'])
                # print()
                # print(theorem_gdl['paras'])
                # print(paras)
                # print('not_added_paras', not_added_paras)
                # print()
                # print()
                while len(not_added_paras) > 0:  # complete the paras
                    para = not_added_paras.pop()
                    theorem_paras.append(para)
                    # print(theorem_name)
                    # print(para)
                    # print(para in theorem_gdl['circle'])
                    # print(self.predicate_to_fact_instances['Circle'])
                    # print(self.predicate_to_fact_instances['Point'])
                    # print()

                    new_instance_to_father_ids = {}
                    if para in theorem_gdl['circle']:  # para is Circle
                        for instance in instance_to_father_ids:
                            for circle in self.predicate_to_fact_instances['Circle']:
                                circle = circle[0]
                                new_instance = list(instance)
                                new_instance.append(circle)
                                new_instance_to_father_ids[tuple(new_instance)] = instance_to_father_ids[instance]
                    else:  # para is Point
                        for instance in instance_to_father_ids:
                            for point in self.predicate_to_fact_instances['Point']:
                                point = point[0]
                                new_instance = list(instance)
                                new_instance.append(point)
                                new_instance_to_father_ids[tuple(new_instance)] = instance_to_father_ids[instance]
                    instance_to_father_ids = new_instance_to_father_ids

                # print(theorem_paras)
                # print()
                # for instance in instance_to_father_ids:
                #     print(instance)
                # print()

                for instance in instance_to_father_ids:
                    father_ids = instance_to_father_ids[instance]
                    # print(instance)
                    replace = dict(zip(theorem_paras, instance))
                    # print(replace)
                    # print()
                    # theorem = theorem_name + '(' + ','.join(replace_paras(theorem_gdl['paras'], replace)) + ')'
                    # print(theorem)

                    if not self._pass_algebraic_constraints(theorem_gdl, replace):  # ac checks
                        continue

                    sub_goals = self._generate_sub_goals(theorem_gdl, replace)  # generate sub_goals
                    if sub_goals is None:
                        continue

                    theorem_instance = replace_paras(theorem_gdl['paras'], replace)
                    for father_id in father_ids:  # add sub_goals
                        operation_id = self._add_operation(('Decompose', theorem_name, theorem_instance))
                        goal_ids = self._add_goals(sub_goals, father_id, operation_id)
                        if goal_ids is not None:
                            update = True
                            all_goal_ids.update(goal_ids)

            else:  # decompose algebraic goal
                if theorem_name not in self.theorem_instances:
                    self._generate_theorem_instances(theorem_name)

                # print(len(self.theorem_instances[theorem_name]))
                # print()
                for theorem, premises, instance in self.theorem_instances[theorem_name]:
                    # print(theorem)
                    # print(premises)
                    # print(instance)
                    # print()
                    father_ids = self._find_father_ids('Eq', instance)  # find father nodes
                    if len(father_ids) == 0:
                        continue

                    _, theorem_instance = parse_fact(theorem)
                    for father_id in father_ids:  # add sub_goals
                        operation_id = self._add_operation(('Decompose', theorem_name, theorem_instance))
                        goal_ids = self._add_goals(premises, father_id, operation_id)
                        if goal_ids is not None:
                            update = True
                            all_goal_ids.update(goal_ids)

        self._check_goals(all_goal_ids)

        return update

    def show(self):
        operation_ids = set()
        goal_related_operation_ids = set()
        if len(self.goals) > 0 and self.status_of_goal[0] == 1:
            goal_related_premise_ids = list(self.premise_ids_of_goal[0])
        else:
            goal_related_premise_ids = []
        for fact_id in goal_related_premise_ids:
            goal_related_operation_ids.add(self.facts[fact_id][3])
            for new_fact_id in self.facts[fact_id][2]:
                if new_fact_id not in goal_related_premise_ids:
                    goal_related_premise_ids.append(new_fact_id)
        goal_related_premise_ids = set(goal_related_premise_ids)

        pf = '{0:<15}{1:<45}{2:<40}{3:<15}{4:<100}'
        pfu = '\033[32m' + pf + '\033[0m'
        for predicate in self.predicate_to_fact_instances:
            if len(self.predicate_to_fact_instances[predicate]) == 0:
                continue

            if predicate in self.parsed_gdl['Presets']:
                print(f'\033[34mPreset - {predicate}:\033[0m')
            else:
                print(f'\033[34mRelation - {predicate}:\033[0m')
            print('\033[34m' + pf.format(
                'fact_id', 'instance', 'premise_ids', 'operation_id', 'operation') + '\033[0m')
            for instance in self.predicate_to_fact_instances[predicate]:
                fact_id = self.fact_id[(predicate, instance)]
                _, _, premise_ids, operation_id = self.facts[fact_id]
                operation_ids.add(operation_id)
                operation = _anti_parse_operation(self.operations[operation_id])
                if predicate != 'Eq':
                    instance = '(' + ','.join(instance) + ')'
                else:
                    instance = str(instance).replace(' ', '')
                premise_ids = '{' + ','.join([str(item) for item in sorted(list(premise_ids))]) + '}'
                if fact_id not in goal_related_premise_ids:
                    print(pf.format(fact_id, instance, premise_ids, operation_id, operation))
                else:
                    print(pfu.format(fact_id, instance, premise_ids, operation_id, operation))
            print()

        if len(self.sym_to_syms) > 0:
            sym_pf = '{0:<50}{1:<15}{2:<50}{3:<10}{4:<20}'
            sym_pfu = '\033[32m' + sym_pf + '\033[0m'
            print('\033[33mAlgebraic System - Symbols:\033[0m')
            print('\033[33m' + sym_pf.format(
                'attribution', 'sym', 'multiple_forms', 'fact_id', 'value') + '\033[0m')
            for sym in self.sym_to_syms:
                if '.' in str(sym):
                    entities, attr = str(sym).split('.')
                    predicate = self.parsed_gdl['Attributions'][attr]['name']
                    instance = ",".join(list(entities))
                    attr = f'{predicate}({instance})'
                    multiple_forms = '(' + ', '.join([str(item) for item in self.sym_to_syms[sym]]) + ')'
                else:
                    attr = f'Free({str(sym)})'
                    multiple_forms = '(' + str(sym) + ')'

                if sym in self.sym_to_value:
                    fact_id = self.fact_id[('Eq', sym - self.sym_to_value[sym])]
                    value = str(self.sym_to_value[sym])
                else:
                    fact_id = 'None'
                    value = "None"

                if fact_id != 'None' and fact_id in goal_related_premise_ids:
                    print(sym_pfu.format(attr, str(sym), multiple_forms, fact_id, value))
                else:
                    print(sym_pf.format(attr, str(sym), multiple_forms, fact_id, value))
            print()

        if len(self.equations) > 0:
            eq_groups_pf = '{0:<10}{1:<45}{2:<35}{3:<15}'
            print('\033[33mAlgebraic System - Equation groups:\033[0m')
            print('\033[33m' + eq_groups_pf.format(
                'group_id', 'simplified_eq', 'premise_ids', 'free_symbols') + '\033[0m')
            for group_id in self.equations:
                for i in range(len(self.equations[group_id][0])):
                    simplified_eq = str(self.equations[group_id][0][i]).replace(' ', '')
                    premise_ids = ','.join([str(item) for item in sorted(list(self.equations[group_id][1][i]))])
                    premise_ids = '{' + premise_ids + '}'
                    free_symbols = ', '.join([str(item) for item in self.equations[group_id][0][i].free_symbols])
                    free_symbols = '(' + free_symbols + ')'
                    print(eq_groups_pf.format(group_id, simplified_eq, premise_ids, free_symbols))
                print()

        if len(self.simplified_algebraic_goal) > 0:
            algebraic_goal_pf = '{0:<10}{1:<45}{2:<35}{3:<15}{4:<30}'
            print('\033[33mAlgebraic System - Algebraic Goals:\033[0m')
            print('\033[33m' + algebraic_goal_pf.format(
                'goal_id', 'simplified_eq', 'premise_ids', 'group_ids', 'dependent_syms', ) + '\033[0m')
            for goal_id in self.simplified_algebraic_goal:
                simplified_eq, premise_ids, dependent_syms, group_ids = self.simplified_algebraic_goal[goal_id]
                simplified_eq = str(simplified_eq).replace(' ', '')
                premise_ids = ','.join([str(item) for item in sorted(list(premise_ids))])
                premise_ids = '{' + premise_ids + '}'
                dependent_syms = ','.join([str(item) for item in sorted(list(dependent_syms), key=str)])
                dependent_syms = '{' + dependent_syms + '}'
                group_ids = ','.join([str(item) for item in sorted(list(group_ids))])
                group_ids = '{' + group_ids + '}'
                print(algebraic_goal_pf.format(goal_id, simplified_eq, premise_ids, group_ids, dependent_syms))
            print()

        goal_pf = '{0:<10}{1:<40}{2:<40}{3:<10}{4:<10}{5:<35}{6:<15}{7:<100}'
        goal_pfs = '\033[32m' + goal_pf + '\033[0m'
        goal_pfu = '\033[31m' + goal_pf + '\033[0m'
        print("\033[35mGoals:\033[0m")
        print('\033[35m' + goal_pf.format('goal_id', 'predicate', 'instance', 'father_id', 'status',
                                          'premise_ids', 'operation_id', 'operation') + '\033[0m')
        last_operation_id = self.goals[0][3]
        for goal_id in range(len(self.goals)):
            predicate, instance, father_id, operation_id = self.goals[goal_id]
            if last_operation_id != operation_id:
                print()
            last_operation_id = operation_id
            if predicate != 'Eq':
                instance = '(' + ','.join(instance) + ')'
            else:
                instance = str(instance).replace(' ', '')
            father_id = str(father_id)
            status = self.status_of_goal[goal_id]
            operation_ids.add(operation_id)
            operation = _anti_parse_operation(self.operations[operation_id])
            if status == 1:
                premise_ids = ','.join([str(item) for item in sorted(list(self.premise_ids_of_goal[goal_id]))])
                premise_ids = '{' + premise_ids + '}'
            else:
                premise_ids = '{}'

            if status == 1:
                print(goal_pfs.format(goal_id, predicate, instance, father_id, status, premise_ids,
                                      operation_id, operation))
            elif status == -1:
                print(goal_pfu.format(goal_id, predicate, instance, father_id, status, premise_ids,
                                      operation_id, operation))
            else:
                print(goal_pf.format(goal_id, predicate, instance, father_id, status, premise_ids,
                                     operation_id, operation))
        print()

        if len(self.theorem_instances) > 0:
            theorem_instance_pf = '{0:<50}{1:<100}'
            print('\033[35mTheorem instances:\033[0m')
            print('\033[35m' + theorem_instance_pf.format('theorem_name', 'theorem_instances') + '\033[0m')
            for t_name in self.theorem_instances:
                # t_instances = [f"({','.join(item)})" for item in self.theorem_instances[t_name][0]]
                # t_instances = f"[{', '.join(t_instances)}]"
                print(theorem_instance_pf.format(t_name, str(self.theorem_instances[t_name])))

        operation_pf = '{0:<15}{1:<50}'
        operation_pfu = '\033[32m' + operation_pf + '\033[0m'
        print('\033[36mOperations:\033[0m')
        print('\033[36m' + operation_pf.format('operation_id', 'operation') + '\033[0m')
        for operation_id in range(len(self.operations)):
            if operation_id not in operation_ids:
                continue
            operation = _anti_parse_operation(self.operations[operation_id])
            if operation_id in goal_related_operation_ids:
                print(operation_pfu.format(operation_id, operation))
            else:
                print(operation_pf.format(operation_id, operation))
        print()

    def state(self):
        serialized_graph = ['<construction>']  # add constructions
        for fact_id in range(len(self.facts)):
            predicate, instance, premise_ids, operation_id = self.facts[fact_id]
            if predicate not in {'Shape', 'Collinear', 'Cocircular'}:
                break
            if serialized_graph[-1] != '<construction>':
                serialized_graph.append('&')
            if predicate == 'Shape':
                serialized_graph.append('Shape')
                for item in instance:
                    if serialized_graph[-1] != 'Shape':
                        serialized_graph.append(',')
                    serialized_graph.extend(list(item))
            elif predicate == 'Collinear':
                serialized_graph.extend(instance)
            else:
                serialized_graph.append(instance[0])
                if len(instance) > 1:
                    serialized_graph.append(',')
                    serialized_graph.extend(instance[1])

        serialized_graph.append('<init_fact>')  # add init facts
        for fact_id in range(len(self.facts)):
            predicate, instance, premise_ids, operation_id = self.facts[fact_id]
            if predicate in {'Shape', 'Collinear', 'Cocircular'}:
                continue
            if len(premise_ids) != 0:
                continue
            if serialized_graph[-1] != '<init_fact>':
                serialized_graph.append('&')
            serialized_graph.extend(_serialize_fact(predicate, instance))

        edges = set()
        for fact_id in range(len(self.facts)):  # add facts
            predicate, instance, premise_ids, operation_id = self.facts[fact_id]
            if predicate in {'Shape', 'Collinear', 'Cocircular'}:
                continue
            if len(premise_ids) == 0:
                continue
            if self.operations[operation_id][1] == 'extend_construction':
                continue
            edge = (tuple(sorted(list(premise_ids))), operation_id,
                    tuple(sorted(list(self.operation_groups[operation_id]))))
            if edge in edges:
                continue
            edges.add(edge)

            serialized_graph.append('<premise>')
            for premise_id in edge[0]:
                predicate, instance, _, _ = self.facts[premise_id]
                if serialized_graph[-1] != '<premise>':
                    serialized_graph.append('&')
                serialized_graph.extend(_serialize_fact(predicate, instance))

            serialized_graph.append('<apply_theorem>')
            serialized_graph.extend(_serialize_operation(self.operations[operation_id]))

            serialized_graph.append('<conclusion>')
            for conclusion_id in edge[2]:
                predicate, instance, _, _ = self.facts[conclusion_id]
                if serialized_graph[-1] != '<conclusion>':
                    serialized_graph.append('|')
                serialized_graph.extend(_serialize_fact(predicate, instance))

        serialized_graph.append('<init_goal>')  # add init goal
        predicate, instance, _, _ = self.goals[0]
        serialized_graph.extend(_serialize_fact(predicate, instance))

        sub_goals = {}  # (goal_id, operation_id): {sub_goal_id}
        for goal_id in range(1, len(self.goals)):
            predicate, instance, father_id, operation_id = self.goals[goal_id]
            if (father_id, operation_id) not in sub_goals:
                sub_goals[(father_id, operation_id)] = {goal_id}
            else:
                sub_goals[(father_id, operation_id)].add(goal_id)

        for goal in sub_goals:
            father_id, operation_id = goal
            serialized_graph.append('<goal>')
            predicate, instance, _, _ = self.goals[father_id]
            serialized_graph.extend(_serialize_fact(predicate, instance))

            serialized_graph.append('<decompose>')
            serialized_graph.extend(_serialize_operation(self.operations[operation_id]))

            serialized_graph.append('<sub_goals>')
            for sub_goal_id in sorted(list(sub_goals[goal])):
                if serialized_graph[-1] != '<sub_goals>':
                    serialized_graph.append('&')
                predicate, instance, _, _ = self.goals[sub_goal_id]
                serialized_graph.extend(_serialize_fact(predicate, instance))

        return serialized_graph
