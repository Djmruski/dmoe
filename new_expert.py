import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

class Expert(nn.Module):
    """
    The Expert model. It has only one hidden layer with hidden_size units.
    """
    
    def __init__(self, input_size=768, hidden_size=20, output_size=2, projected_output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.mapper = nn.Linear(in_features=output_size, out_features=projected_output_size, bias=False)
        self.batchnorm = nn.InstanceNorm1d(num_features=hidden_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = F.relu(self.batchnorm(self.fc1(x)))
        out = self.mapper(self.fc2(out))
        return out

class BiasLayer(torch.nn.Module):
    """
    The bias layer adapted from BiC.
    It will be added to the end of expert classification layers.
    It has only two parameters: alpha and beta
    """
    
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x):
        return self.alpha * x + self.beta        

class DynamicExpert(nn.Module):
    """
    The DynamicExpert model.
    It will add/create a new expert when the new task comes.
    """
    
    def __init__(self, input_size=768, hidden_size=20, total_cls=100):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.total_cls = total_cls
        self.gate = None
        self.experts = None
        self.bias_layers = None
        self.prev_classes = []
        self.cum_classes = set()
        self.relu = nn.ReLU()


    def expand_expert(self, seen_cls, new_cls):
        """
        Expand existing expert given the seen classes and the new classes
        """
        
        self.seen_cls = seen_cls
        self.new_cls = new_cls

        if not self.experts:
            """
            If there is no expert yet, i.e. this is the first task, then create the gate, expert and bias layer as usual
            """
            self.prev_classes.append(self.new_cls)
            gate = nn.Linear(in_features=self.input_size, out_features=1)
            experts = nn.ModuleList([Expert(input_size=self.input_size, hidden_size=self.hidden_size, output_size=new_cls, projected_output_size=new_cls)])
            self.bias_layers = nn.ModuleList([BiasLayer()])
            self.num_experts = len(experts)            

            # task token
            token = nn.Parameter(torch.zeros(1, 1, self.input_size))
            nn.init.trunc_normal_(token, std=.02)
            self.task_tokens = nn.ParameterList([token])
        else:            
            """
            If there is already an expert:
                - create a new gate (i.e. the previou gate will be discarded)
                - create a new expert and adjust the class mapper
                - add a new bias layer to the existing list
            """
            self.prev_classes.append(self.new_cls)
            gate = nn.Linear(in_features=self.input_size, out_features=self.num_experts+1)                  
            experts = copy.deepcopy(self.experts)
            experts.append(Expert(input_size=self.input_size, hidden_size=self.hidden_size, output_size=new_cls, projected_output_size=new_cls))
            self.num_experts = len(experts)
            for expert_index, module in enumerate(experts):
                start = sum(self.prev_classes[:expert_index])
                end = start + self.prev_classes[expert_index]

                weight = module.mapper.weight
                input_size = module.mapper.in_features
                new_mapper = nn.Linear(in_features=input_size, out_features=sum(self.prev_classes), bias=False)

                with torch.no_grad():
                    all_ = {i for i in range(sum(self.prev_classes))}
                    kept_ = {i for i in range(start, end)}
                    removed_ = all_ - kept_
                    
                    upper_bound = sum(self.prev_classes[:expert_index+1])

                    new_mapper.weight[start:end, :] = weight if weight.size(0) <= new_cls else weight[start:upper_bound, :]
                    new_mapper.weight[list(removed_)] = 0.
                    module.mapper = new_mapper

            self.bias_layers.append(BiasLayer())

            # task token
            new_task_token = copy.deepcopy(self.task_tokens[-1])
            nn.init.trunc_normal_(new_task_token, std=.02)
            self.task_tokens.append(new_task_token)
        
        self.gate = gate
        self.experts = experts
        # self.bn1 = nn.BatchNorm1d(num_features=gate.out_features)
        # self.in1 = nn.InstanceNorm1d(num_features=gate.out_features)

        # gate_total_params = sum(p.numel() for p in gate.parameters())
        # print(f"task-{self.num_experts-1} GATE_TOTAL_PARAMS: {gate_total_params}")
        # expert_total_params = sum(p.numel() for p in experts.parameters())
        # print(f"task-{self.num_experts-1} EXPERT_TOTAL_PARAMS: {expert_total_params}")
        # bias_total_params = sum(p.numel() for p in self.bias_layers.parameters())
        # print(f"task-{self.num_experts-1} bias_TOTAL_PARAMS: {bias_total_params}")                    

    def calculate_gate_norm(self):
        """
        Calculate the norm of the gate.
        Not used during training, only for investigating.
        """
        
        w1 = nn.utils.weight_norm(self.gate, name="weight")
        print(w1.weight_g)
        nn.utils.remove_weight_norm(w1)

    def bias_forward(self, task, output):
        """Modified version from FACIL"""
        return self.bias_layers[task](output)

    def freeze_previous_experts(self):        
        for i in range(len(self.experts) - 1):
            e = self.experts[i]
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all_experts(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all_bias(self):
        for b in self.bias_layers:
            b.alpha.requires_grad = False
            b.beta.requires_grad = False

    def unfreeze_all_bias(self):
        for b in self.bias_layers:
            b.alpha.requires_grad = True
            b.beta.requires_grad = True

    def set_gate(self, grad):
        for name, param in self.named_parameters():
            if name == "gate":
                param.requires_grad = grad

    def unfreeze_all(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = True

    def forward(self, x, task=None, train_step=2):
        """
        The training step when xs (samples) are given.
        In step 1, only expert is trained.
        In step 2, only gate is trained (and bias layers)
        """
        gate_outputs = None
        if train_step == 1:            
            expert_outputs = self.experts[task](x)
        else:
            # if x.size(0) != 1:
            #     gate_outputs = self.relu(self.bn1(self.gate(x)))
            # else:
            #     gate_outputs = self.relu(self.in1(self.gate(x)))
            gate_outputs = self.gate(x)
            # gate_outputs = self.relu(self.gate(x))

            # gate_outputs = self.relu(self.gate(x))
            gate_outputs_uns = torch.unsqueeze(gate_outputs, 1)
            
            expert_outputs = [self.experts[i](x) for i in range(self.num_experts)]
            # print(f"in new_expert 1: {len(expert_outputs)}")
            expert_outputs = torch.stack(expert_outputs, 1)
            # print(f"in new_expert 2: {expert_outputs.size()}")
            expert_outputs = gate_outputs_uns@expert_outputs
            # print(f"in new_expert 3: {expert_outputs.size()}")
            expert_outputs = torch.squeeze(expert_outputs, 1) # only squeeze the middle 1 dimension
            # print(f"in new_expert 4: {expert_outputs.size()}")

        return expert_outputs, gate_outputs

    def predict(self, x, task):
        expert_output = self.experts[task](x)
        return expert_output
        