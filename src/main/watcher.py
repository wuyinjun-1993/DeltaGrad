'''
Created on Oct 22, 2018

'''
import sys
import traceback

from main.Parse_python_file import parse_update_statement
from main.Parse_python_file import tree_assignment
import torch
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


class Watcher(object):

    
    
    def add_annotation2variables(self, annotate_input_by_row):
        
        shape = list(self.input_data.size())
        if sample_level:
            prov = M_prov.add_prov_token(self.input_data, self.input_data_name, shape, annotate_input_by_row)
        
        else:
            prov = M_prov.add_prov_token(self.input_data, self.input_data_name, shape)
        
        print('provenance of input')
        print(prov.prov_list)
        print(prov.data_matrix_list)
        
        for key in self.parsed_update_sts:
            parse_tree = self.parsed_update_sts[key]
            parse_tree.iterate_tree_add_provenance(self.input_data_name, prov)
    
    
    def __init__(self, output_var_name = None, obj=None, attr=None, input_data=None, input_data_name = None, log_file='log.txt', include=[], enabled=False, file_name=None, annotate_input_by_row = True, annotate_para_by_row = True):
        """
            Debugger that watches for changes in object attributes
            obj - object to be watched
            attr - string, name of attribute
            log_file - string, where to write output
            include - list of strings, debug files only in these directories.
               Set it to path of your project otherwise it will take long time
               to run on big libraries import and usage.
        """


        self.log_file=log_file
        self.output_var_name = output_var_name
        with open(self.log_file, 'wb'): pass
        self.prev_st = None
        self.include = [incl.replace('\\','/') for incl in include]
        self.value = None
        if obj:
            self.value = getattr(obj, attr)
        self.obj = obj
        self.attr = attr
        self.enabled = enabled # Important, must be last line on __init__.
        self.input_data = input_data
        self.input_data_name = input_data_name
        self.parsed_update_sts = parse_update_statement(file_name)
        self.add_annotation2variables(annotate_input_by_row)
        self.output_prov = None
        if not self.value is None:
            
            if sample_level:
                self.output_prov = M_prov.add_prov_token_constants(self.value, list(self.value.size()), annotate_para_by_row)
            else:
                self.output_prov = M_prov.add_prov_token_constants(self.value, list(self.value.size()))

    def __call__(self, *args, **kwargs):
        kwargs['enabled'] = True
        self.__init__(*args, **kwargs)

    def check_condition(self):
        
#         print(self.attr, self.value)
        
        tmp = getattr(self.obj, self.attr)
        result = not tmp.equal(self.value)
        self.value = tmp
        return result

    def trace_command(self, frame, event, arg):
        if event !='line' or not self.enabled:
            return self.trace_command
        
        
        if self.check_condition():
#             print('here!!!!!!!!!!')
#             
#             print(event)
#             print(frame.f_locals)
#             print(arg)
            
            if self.output_prov is None:
                self.output_prov = M_prov.add_prov_token_constants(self.value, list(self.value.size()))
#             print('123', self.prev_st)
            if self.prev_st:
                
                
#                 print('here!!!!!!!!!!')
#                 print(event)
#                 print(frame.f_locals)
#                 print(arg)
# #                 with open(self.log_file, 'ab') as f:
#                 print('obj',self.obj)
#                 print('attr',self.attr)
#                 print('value', self.value)
#                 print('statement', self.prev_st)
                
                sts_len = len(self.prev_st)
                sts = self.prev_st[sts_len - 1].split('\n')[0].split(',')[1].split(' ')[2]
                lineno = int(sts.strip())
                curr_tree = self.parsed_update_sts[lineno]
                self.output_prov = curr_tree.post_order_traverse_compute_prov(variable_map = frame.f_locals, tracking_variable_name = self.output_var_name, tracking_varible_prov = self.output_prov)
                print('statistics::')
                self.return_maximal_tensor_product(self.output_prov)
#                 print('here', curr_tree.root.value)
#                 self.compute_res(self.output_prov)
#                     print("Value of",self.obj,".",self.attr,"changed!", file = f)
#                     print(self.obj, file=f)
#                     print("###### Line:", file=f)
#                     print(''.join(self.prev_st), file = f)
        if self.include:
            fname = frame.f_code.co_filename.replace('\\','/')
#             print(fname)
            to_include = False
            for incl in self.include:
                if fname.startswith(incl):
                    to_include = True
                    break
            if not to_include:
                return self.trace_command
        self.prev_st = traceback.format_stack(frame)
        
        return self.trace_command
    
    
    def return_maximal_tensor_product(self, P):
        
        max_value = torch.tensor(0, dtype = torch.float32)
        
        max_row = -1
        
        max_column = -1
        
        max_id = -1
        
        
        for i in range(len(P.data_matrix_list)):
            for j in range(len(P.data_matrix_list[0])):
                curr_data_matrix = P.data_matrix_list[i][j]
                
                for k in range(len(curr_data_matrix)):
                    if torch.norm(curr_data_matrix[k]) > torch.norm(max_value):
                        max_value = curr_data_matrix[k]
                        max_row = i
                        max_column = j
                        max_id = k
                 
        print(max_value, P.prov_list[max_row][max_column][max_id], P.prov_exponent_list[max_row][max_column][max_id])
    
    
    def compute_res(self, P):
        
        res = []
        
        for i in range(len(P.data_matrix_list)):
            curr_res = []
            for j in range(len(P.data_matrix_list[0])):
                
                this_res = 0
                
                curr_data_matrix_list = P.data_matrix_list[i][j]
                
                curr_prov_list = P.prov_list[i][j]
                
                for k in range(len(curr_data_matrix_list)):
                    if not "X'0,0" in curr_prov_list[k]:
                        this_res += curr_data_matrix_list[k]
                
                curr_res.append(this_res) 
                
            res.append(curr_res)
        print('result::', res)
# import sys
# watcher = Watcher()
# sys.settrace(watcher.trace_command)