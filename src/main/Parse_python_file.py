'''
Created on Oct 22, 2018

'''
import ast
from builtins import getattr
from collections import deque
import logging
import os

import torch


# from main.matrix_prov_sample_level import M_prov
# from main.matrix_prov_sample_level import M_prov
sample_level = False
full_provenance = True


if full_provenance:
    from main.matrix_prov_entry_level_full import M_prov
else:
    if sample_level:
        from main.matrix_prov_sample_level import M_prov
    else:
        from main.matrix_prov_entry_level import M_prov

# from torch import tensor

class node_assignment:
    
    var_node = "var"
    const_node = "var"
    op_node = "op"
    add_op = {"Add()", "torch.add"}
    matrix_mul_op = {"torch.mm"}
    mul_op = {"Mult()"}
    neg_op = {"torch.neg"}
    sub_op = {"Sub()"}
    transpose = {"torch.transpose"}
    
    '''value: value'''
    def __init__(self, value, parent, attr = None, type = None):
        
        self.parent = parent
        self.children = []
        
        if not parent is None:
            parent.children.append(self)
        
        
        self.value = value
        self.attr = attr
        self.type = type 
#         if not parent is None:
#             print('node', value, parent.value)
#         if isinstance(value, ast.Name):
#             self.v = value
#         
#         if isinstance(value, ast.operator):
#             self.op = value
        
        self.prov_expr = None

class tree_assignment:
    
    '''todo: parse a[0][0].b[1][1]'''
    def __init__(self, node=None):
        root_element = []
        for elt in node.targets:
            if hasattr(elt, 'id'):
                root_element.append(elt.id)
            else:
                print(ast.dump(elt))
                if isinstance(elt, ast.Subscript):
                    value = self.process_subscript(elt)
                    print('element', ast.dump(elt))
                    root_element.append(value)
                    print('value::' + value)
                else:
#                 print(ast.dump(elt.value))
                    root_element.append(elt.value.id + "." + elt.attr)
        self.target = root_element
        
        print(root_element)
#         if node.value
        
        root = self.construct_tree(node.value, None)
        
#         root_val = None
#         
#         if isinstance(node.value, ast.BinOp) or isinstance(node.value, ast.UnaryOp):
#             root_val = str(ast.dump(node.value.op))
#             
#             
#         
#         if isinstance(node.value, ast.Call):
#             if hasattr(node.value.func, 'id'):
#                 root_val = node.value.func.id
#             else:
#                 print(ast.dump(node.value.func.value))
#                 root_val = node.value.func.value.id + "." + node.value.func.attr
#         
#         print(root_val)
#         
#         
#         
#         root = node_assignment(root_val, None)
        
        self.root = root
        
    def process_list(self, node):
        l = []
        for elt in node.elts:
            if isinstance(elt, ast.Num):
                l.append(elt.n)
            if isinstance(elt, ast.List):
                l.append(self.process_list(elt))
                
            if isinstance(elt, ast.Str):
                l.append(elt.s)
            
            if isinstance(elt, ast.Subscript):
                l.append(elt.value.id + '[' + str(elt.slice.value.n) + ']')
        return l
    
    def process_subscript(self, node):
        
        stack = []
        curr_node = node
        
        while isinstance(curr_node.value, ast.Subscript) or isinstance(curr_node.value, ast.Attribute):
            print(ast.dump(curr_node.value))
            if isinstance(curr_node, ast.Subscript):
                stack.append('[' + str(curr_node.slice.value.n) + ']')
            else:
                stack.append('.' + str(curr_node.attr))
            curr_node = curr_node.value
        value = curr_node.value.id
        
        stack.append('[' + str(curr_node.slice.value.n) + ']')
        
        while len(stack) > 0:
            value += stack.pop()
        
        return value
            
    
    def process_basic_data_type(self, node):
        if isinstance(node, ast.List):
            value = self.process_list(node)
        
        if isinstance(node, ast.Subscript):
            value = self.process_subscript(node)
        
        if isinstance(node, ast.Name):
            value = node.id
        if isinstance(node, ast.Num):
            value = node.n
        if isinstance(node, ast.Str):
            value = node.s
        if isinstance(node, ast.NameConstant):
            value = node.value
            
        if isinstance(node, ast.Attribute):
            value = self.process_basic_data_type(node.value) + '.' + node.attr
        
#         print(ast.dump(node))
        return value
    
    def construct_tree(self, node, root, attr = None):
        if isinstance(node, ast.List):
            l_node = node_assignment(self.process_list(node), root, attr, node_assignment.var_node)
        
        if isinstance(node, ast.NameConstant):
            l_node = node_assignment(node.value, root, attr, node_assignment.const_node)
        
        if isinstance(node, ast.Subscript):
            l_node = node_assignment(self.process_subscript(node), root, attr, node_assignment.var_node)
        
        if isinstance(node, ast.Name):
            l_node = node_assignment(node.id, root, attr, node_assignment.var_node)
        if isinstance(node, ast.Num):
            l_node = node_assignment(node.n, root, attr, node_assignment.const_node)
        if isinstance(node, ast.Str):
            l_node = node_assignment(node.s, root, attr, node_assignment.const_node)
            
#         if isinstance(node, ast.keyword):
#             l_node = node_assignment(self.process_basic_data_type(node.value), root, node.arg)
        
        if isinstance(node, ast.Attribute):
            l_node = node_assignment(self.process_basic_data_type(node.value) + "." + node.attr, root, attr, node_assignment.var_node)    
        
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'id'):
                l_node = node_assignment(node.func.id, root, attr, node_assignment.op_node)
            else:
                l_node = node_assignment(self.process_basic_data_type(node.func.value) + "." + node.func.attr, root, attr, node_assignment.op_node)
            for arg in node.args:
                self.construct_tree(arg, l_node)
            
            for arg in node.keywords:
                self.construct_tree(arg.value, l_node, arg.arg)
        
        if isinstance(node, ast.BinOp):
            l_node = node_assignment(ast.dump(node.op), root, attr, node_assignment.op_node)
            left = node.left
            self.construct_tree(left, l_node)
            right = node.right
            self.construct_tree(right, l_node)
        
        
        if isinstance(node, ast.UnaryOp):
            l_node = node_assignment(ast.dump(node.op), root, attr, node_assignment.op_node)
            left = node.operand
            self.construct_tree(left, l_node)
        return l_node
    
    def iterate_tree(self):
        queue = deque([])
        root = self.root
        queue.append(root)
        while len(queue) > 0:
            node = queue.popleft()
            logging.debug('node', node.value)
            for child in node.children:
                logging.debug('child', child.value)
                queue.append(child)
    
    def iterate_tree_add_provenance(self, v_name, Prov):
        queue = deque([])
        root = self.root
        queue.append(root)
        while len(queue) > 0:
            node = queue.popleft()
            if node.value == v_name:
                node.prov_expr = Prov
                
            
            
            logging.debug('node', node.value)
            for child in node.children:
                logging.debug('child', child.value)
                queue.append(child)
    def print_children(self, root):
        print('children:')
        for child in root.children:
            print(child.value)
            print(child.prov_expr.prov_list)
            if full_provenance:
                print(child.prov_expr.prov_exponent_list)
            if child.value == 'lr.theta':
                print(child.prov_expr.data_matrix_list)
                
    def print_intermmediated_result(self, root):
        sum = []
        
        for i in range(len(root.prov_expr.data_matrix_list)):
            curr_sum = root.prov_expr.data_matrix_list[i][0]
            for j in range(len(root.prov_expr.data_matrix_list[i])-1):
                curr_sum = curr_sum + root.prov_expr.data_matrix_list[i][j+1]
            sum.append(curr_sum)
        print('inter_data', root.prov_expr.data_matrix_list)
        print('inter_sum', sum)
    
    def compute_exponent(self, prov_exponent_list):
        max_exponent = -1
        
        for i in range(len(prov_exponent_list)):
            for j in range(len(prov_exponent_list[i])):
                for k in range(len(prov_exponent_list[i][j])):
#                     for p in range(len(root.prov_expr.prov_exponent_list[i][j][k])):
                        curr_exponent = sum(prov_exponent_list[i][j][k])
                        
                        max_exponent = max(max_exponent, curr_exponent)
        return max_exponent
    
    def print_root_res(self, root):
        print('root_res:')
        print(root.prov_expr.prov_list)
        if full_provenance:
            print(root.prov_expr.prov_exponent_list)
        
            print('max_exponent', self.compute_exponent(root.prov_expr.prov_exponent_list))
        
        
        
    
    
    def post_order_traverse(self, root, variable_map, tracking_variable_name, tracking_varible_prov):
        
        for node in root.children:
            self.post_order_traverse(node, variable_map, tracking_variable_name, tracking_varible_prov)
            
        '''compute the provenance for root'''
        if root.type == node_assignment.op_node:
            if root.value in node_assignment.add_op:
                root.prov_expr = root.children[0].prov_expr
                for i in range(len(root.children) - 1):
                    node = root.children[i+1]
                    root.prov_expr = M_prov.prov_matrix_add_prov_matrix(root.prov_expr, node.prov_expr, None)
#                 print('add()')
#                 self.print_children(root)
#                 self.print_intermmediated_result(root)
#                 print('root', root.value)
#                 print('prov', root.prov_expr.prov_list)
            if root.value in node_assignment.matrix_mul_op:
                root.prov_expr = root.children[0].prov_expr
#                 if root.children[0].value == 'alpha':
#                     print('alpha_prov', root.children[0].prov_expr.prov_list)
#                 print('root', root.value)
#                 print('here', root.children[0].value)
#                 print('here',root.children[0].prov_expr.prov_list)
#                 print('here', root.children[0].prov_expr.data_matrix_list)
                for i in range(len(root.children) - 1):
#                     print('here', root.children[i+1].value)
#                     print('here_root', root.value)
#                     print('here',root.children[i+1].prov_expr.prov_list)
#                     print('here', root.children[i+1].prov_expr.data_matrix_list)
#                     print('here', root.prov_expr.prov_list)
#                     print('here', root.prov_expr.data_matrix_list)
#                     if full_provenance:
#                         print('exponent1', self.compute_exponent(root.prov_expr.prov_exponent_list))
#                         print('exponent2', self.compute_exponent(root.children[i+1].prov_expr.prov_exponent_list))

                    root.prov_expr = M_prov.prov_matrix_mul_prov_matrix(root.prov_expr, root.children[i+1].prov_expr, None)
#                 print('matrix_mul()')
#                 self.print_children(root)
#                 self.print_intermmediated_result(root)
#                 self.print_root_res(root)
#                 print('root', root.value)
#                 print('prov', root.prov_expr.prov_list)
            
            if root.value in node_assignment.transpose:
#                 print('transpose', root.children[0].value, root.children[0])
                root.prov_expr = M_prov.prov_matrix_transpose(root.children[0].prov_expr, None)
#                 print('transpose')
#                 self.print_children(root)
#                 self.print_intermmediated_result(root)
#                 print('root', root.value)
#                 print('prov', root.prov_expr.prov_list)
#                 print('data', root.prov_expr.data_matrix_list)
            
            
            if root.value in node_assignment.sub_op:
#                 print('here', root.children[0].value)
#                 print('here', root.children[1].value)
#                 print('here', root.children[0].prov_expr.prov_list)
#                 print('here', root.children[1].prov_expr.prov_list)
                root.prov_expr = M_prov.prov_matrix_add_prov_matrix(root.children[0].prov_expr, M_prov.negate_prov_matrix(root.children[1].prov_expr, None), None)
#                 print('sub()')
#                 self.print_children(root)
#                 self.print_intermmediated_result(root)
#                 self.print_root_res(root)
#                 print('root', root.value)
#                 print('prov', root.prov_expr.prov_list)
#                 if full_provenance:
#                     print('prov_exponent', root.prov_expr.prov_exponent_list)
                
            if root.value in node_assignment.mul_op:
                root.prov_expr = root.children[0].prov_expr
#                 if root.children[0].value == 'alpha':
#                     print('alpha_prov', root.children[0].prov_expr.prov_list)
#                     if full_provenance:
#                         print('alpha_prov', root.children[0].prov_expr.prov_exponent_list)
#                 print('root', root.value)
#                 print('here', root.children[0].value)
#                 print('here',root.children[0].prov_expr.prov_list)
#                 print('here', root.children[0].prov_expr.data_matrix_list)
                for i in range(len(root.children) - 1):
                    
#                     print(i)
#                     
#                     print('here', root.children[i+1].value)
#                     print('here_mul', root.value)
#                     print('here',root.children[i+1].prov_expr.prov_list)
#                     print('here', root.children[i+1].prov_expr.data_matrix_list)
#                     print('here', root.prov_expr.prov_list)
#                     print('here', root.prov_expr.data_matrix_list)
#                     if full_provenance:
#                         print('here!!!!!', root.children[i+1].prov_expr.prov_exponent_list)
#                         print('here', root.prov_expr.prov_exponent_list)
                    root.prov_expr = M_prov.constant_mul_prov_matrix(root.prov_expr, root.children[i+1].prov_expr, None)
#                 print('mul()')
#                 self.print_children(root)
#                 self.print_intermmediated_result(root)
#                 print('root', root.value)
#                 print('prov', root.prov_expr.prov_list)
            
            
        else:
#             print('var_mapping:')
#             print(variable_map)
#             
#             for key in variable_map:
#                 print(variable_map[key].theta)
            curr_var_name = root.value
#             print(curr_var_name)
            if isinstance(curr_var_name, str) and '.' in curr_var_name:
                names = curr_var_name.split('.')
                for i in range(len(names)):
                    sub_name = names[i].split('[')
                    if i == 0:
                        value = variable_map[sub_name[0]]
                    else:
#                         print('subname', sub_name[0])
                        value = getattr(value, sub_name[0])
                    for j in range(len(sub_name) - 1):
                        id = int(sub_name[j+1].split(']')[0])
                        value = value[id]
                
                if root.value == tracking_variable_name:
                        root.prov_expr = tracking_varible_prov
#                         print('curr_theta', value)
#                         print('curr_theta_prov', tracking_varible_prov.data_matrix_list)
                else:
                    if root.prov_expr == None:
                        if not isinstance(value, torch.Tensor):
                            v_l = []
                            v_l.append(value)
                            v_ls = []
                            v_ls.append(v_l)
                            value = torch.tensor(v_ls)
                        root.prov_expr = M_prov.add_prov_token_constants(value, list(value.size()))
                
            else:
                
                try:
                    value = variable_map[curr_var_name]
                    if root.value == tracking_variable_name:
                        root.prov_expr = tracking_varible_prov
                    else:
                        if root.prov_expr == None:
                            if not isinstance(value, torch.Tensor):
                                v_l = []
                                v_l.append(value)
                                v_ls = []
                                v_ls.append(v_l)
                                value = torch.tensor(v_ls)
                            root.prov_expr = M_prov.add_prov_token_constants(value, list(value.size()))
                
                except KeyError:
                    value = root.value
                    if not isinstance(value, torch.Tensor):
                        v_l = []
                        v_l.append(value)
                        v_ls = []
                        v_ls.append(v_l)
                        value = torch.tensor(v_ls)
                    
                    root.prov_expr = M_prov.add_prov_token_constants(value, list(value.size()))
#             
    '''variable_map: variable_name->value of variable'''
    def post_order_traverse_compute_prov(self, variable_map = None, tracking_variable_name = None, tracking_varible_prov = None):
        self.post_order_traverse(self.root, variable_map, tracking_variable_name, tracking_varible_prov)
        print(self.root.prov_expr.prov_list)
        print(self.root.prov_expr.data_matrix_list)
        if full_provenance:
            print(self.root.prov_expr.prov_exponent_list)
        
        sum_matrix = self.root.prov_expr.data_matrix_list[0][0]
        for i in range(len(self.root.prov_expr.data_matrix_list[0]) - 1):
            sum_matrix = torch.add(sum_matrix, self.root.prov_expr.data_matrix_list[0][i + 1])
            
#         print(sum_matrix)
        return self.root.prov_expr
#     def construct_tree(self, node, root):
#         if isinstance(node.value, ast.BinOp):
#             root_val = str(ast.dump(node.value.op))
#             left = node.value.left
#             construct_tree(self, left, root)
#             right = node.value.right
#             
#         
#         if isinstance(node.value, ast.UnaryOp):
#             
#             
#             
#         
#         if isinstance(node.value, ast.Call):
#             if hasattr(node.value.func, 'id'):
#                 root_val = node.value.func.id
#             else:
#                 print(ast.dump(node.value.func.value))
#                 root_val = node.value.func.value.id + "." + node.value.func.attr


def parse_update_statement(file_name):
    base_path = os.path.dirname(__file__)
    fullpath = os.path.join(base_path, file_name)
    data = open(fullpath).read()
    tree = ast.parse(data, fullpath)
    
    node_stack = []
    
    node_stack.append(tree)
    
    map={}
    
    while len(node_stack) > 0:
        curr_node = node_stack.pop()
        for node in ast.iter_child_nodes(curr_node):
            print(ast.dump(node))
            if isinstance(node, ast.Assign):
                logging.debug(node.lineno, ",", ast.dump(node.targets[0]))
                logging.debug(node.lineno, "," , ast.dump(node.value))
                map[node.lineno] = tree_assignment(node)
                map[node.lineno].iterate_tree()
            if isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef) or isinstance(node, ast.For) or isinstance(node, ast.While):
                node_stack.append(node)
            
    
    
#     for key in map:
# #         if isinstance(map[key].value, ast.BinOp):  
#         print(key, ",", ast.dump(map[key].targets[0])) 
#         print(key, ",", ast.dump(map[key].value)) 
# #         if key != 19 and key != 23 and key!= 59 and key != 61:
#         tree = tree_assignment(map[key])
#         tree.iterate_tree()
        

    return map

if __name__ == '__main__':
    parse_update_statement("linear_regression.py")


# for node in ast.iter_child_nodes(tree):
#     if isinstance(node, ast.Assign):
#         
#     
#     if isinstance(node, ast.ClassDef):
#         for n in ast
#     
#     print(ast.dump(node))
#     print(node.lineno)
#     if isinstance(node, ast.FunctionDef):
#         for n2 in ast.iter_child_nodes(node):
#             if isinstance(n2, ast.For):
#                 for n3 in ast.iter_child_nodes(n2):
#                     if isinstance(n3, ast.Assign):
#                         print(ast.dump(n3))
#                         print(n3.lineno)
#                         for n4 in ast.iter_child_nodes(n3):
#                             print(ast.dump(n4))
#                             print(n4.lineno)

# print(ast.dump(tree))

# print(tree)
# print(tree.body)
# 
# for i in range(len(tree.body)):
#     print(ast.Expression(tree.body[i]))


# test_case_path = os.path.join(base_path, "test_cases")
# test_case_files = os.listdir(test_case_path)
# 
# test_cases = []
# 
# for fname in test_case_files:
#     if not fname.endswith(".py"):
#         continue
# 
#     fullpath = os.path.join(test_case_path, fname)
#     data = open(fullpath).read()
#     tree = ast.parse(data, fullpath)
# #     codes, messages = extract_expected_errors(data)
# 
# #     test_cases.append((tree, fullpath, codes, messages))
# 
# 
# 
# import py_compile
# mod = py_compile.compile("linear_regression.py")
# print(mod)