import json
import os
import sys

import numpy as np
import copy
import isaacgym
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(dir_path + "/..")
sys.path.append(dir_path + "/../common")

from common.torch_jit_utils import *
from run import get_agent

model_folder = "logs/rmamtsac/Quadcopter/2025-02-24-12-38-46"
model_checkpoint = "phase1_best"


cfg_path = model_folder + "/cfg"
model_path = model_folder + "/" + model_checkpoint + "/"

cfg_dict = None
with open(cfg_path) as f:
    cfg_dict = json.load(f)

cfg_dict["env"]["num_envs"] = 1
cfg_dict["agent"]["save_model"] = False

agent = get_agent(cfg_dict)
agent.load_torch_model(model_path)

print(agent.policy)
total_params = sum(p.numel() for p in agent.policy.parameters())
model_size = total_params*4/1000
print("number of parameters:", total_params, ", size [KB]:", model_size)
if model_size > 70:
    print("[Warning] model size may exceed flight controller memory limits 70 [KB].")

agent.adaptor = agent.adaptor.eval().to("cpu")
agent.policy = agent.policy.to("cpu")

N_act = agent.action_dim
N_obs = agent.state_dim
N_task = agent.feature_dim
N_latent = agent.latent_dim

pos_lim = agent.env.pos_lim
vel_lim = agent.env.vel_lim
avel_lim = agent.env.avel_lim
authority = agent.env.control_authority

# parameters
N_hidden = cfg_dict["agent"]["policy_net_kwargs"]["hidden_dim"]
N_context = cfg_dict["agent"]["policy_net_kwargs"]["context_dim"]

N_tcn_layers = cfg_dict["agent"]["adaptor_net_kwargs"]["num_tcn_layers"]
N_tcn_hidden = cfg_dict["agent"]["adaptor_net_kwargs"]["hidden_dim"]
N_kernel = cfg_dict["agent"]["adaptor_net_kwargs"]["kernel_size"]
N_padding = cfg_dict["agent"]["adaptor_net_kwargs"]["kernel_size"]-1
N_stack = agent.stack_size

# adaptor
def weight_norm(weight_v, weight_g):
    return weight_g * (weight_v / torch.norm(weight_v, dim=(1,2), keepdim=True))

d = agent.adaptor.state_dict()
print("adaptor net:", d.keys())
cnn_b1 = d['tcn.tcn.network.0.conv1.bias'] # [N_obs]
cnn_c1_wg = d['tcn.tcn.network.0.conv1.weight_g'] # [N_obs, 1, 1]
cnn_c1_wv = d['tcn.tcn.network.0.conv1.weight_v'] # [N_obs, N_obs, N_kernel]
# cnn_n1_b = d['tcn.tcn.network.0.net.0.bias']
# cnn_n1_wg = d['tcn.tcn.network.0.net.0.weight_g']
# cnn_n1_wv = d['tcn.tcn.network.0.net.0.weight_v']
cnn_w1 = weight_norm(cnn_c1_wv, cnn_c1_wg) # [N_obs, N_obs, N_kernel]

if N_tcn_layers>1:
    cnn_b2 = d['tcn.tcn.network.1.conv1.bias'] # [N_obs]
    cnn_c2_wg = d['tcn.tcn.network.1.conv1.weight_g'] # [N_obs, 1, 1]
    cnn_c2_wv = d['tcn.tcn.network.1.conv1.weight_v'] # [N_obs, N_obs, N_kernel]
    cnn_w2 = weight_norm(cnn_c2_wv, cnn_c2_wg)  

cnn_l_w = d['tcn.linear.weight'] # [N_latent, N_obs] 
cnn_l_b = d['tcn.linear.bias'] # [N_latent] 

cnn_lin_w = d['fcn.lin.layer.weight'] # [N_tcn_hidden, N_latent]
cnn_lin_b = d['fcn.lin.layer.bias'] # [N_tcn_hidden]

cnn_lout_w = d['fcn.lout.layer.weight'] # [N_latent, N_tcn_hidden]
cnn_lout_b = d['fcn.lout.layer.bias'] # [N_latent]


# policy
d = agent.policy.state_dict()
print("policy net:", d.keys())

win_w = d["model.w_in.layer.weight"]  # [N_hidden, N_obs+N_latent+N_task] 
win_b = d["model.w_in.layer.bias"]  # [N_hidden] 

wout_w = d["model.w_out.layer.weight"]  # [N_context, N_hidden]  
wout_b = d["model.w_out.layer.bias"]  # [N_context] 

lin_w = d["model.l_in.weight"]  # [N_context, N_hidden, N_obs+N_latent] 
lin_b = d["model.l_in.bias"]  # [N_context, N_hidden]

# l0_w = d["model.layers.0.weight"]  # [N_context, N_hidden, N_hidden] 
# l0_b = d["model.layers.0.bias"]  # [N_context, N_hidden]

mean_w = d["model.mean_linear.weight"]  # [N_context, N_act, N_hidden]
# mean_b = d["model.mean_linear.bias"]  # [N_context, N_act]

# test in numpy
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def linear_layer(bias, weight, input, activation=True, type="relu"):
    # weight: [outdim, indim]
    # bias: [outdim]
    # input: [indim]
    # output: [outdim]
    output = bias + np.dot(weight, input)

    if activation and type == "relu":
        return np.maximum(0, output)
    elif activation and type == "softmax":
        return softmax(output)
    else:
        return output

def composition_layer(bias, weight, comp_weight, input, activation=True):
    # weight: [contextdim, outdim, indim]
    # bias: [contextdim, outdim]
    # input: [indim]
    # comp_weight: [contextdim]
    # output: [outdim]
    x = np.dot(weight, input)  # [contextdim, outdim]
    if bias is not None:
        x += bias  # [contextdim, outdim]

    output = np.dot(comp_weight, x)  # [outdim]

    if activation:
        return np.maximum(0, output)
    else:
        return output


def chomp1d(input, padding):
    return input[:,:-padding]

def conv1d_numpy(input, weight, bias=None, stride=1, padding=0, dilation=1):
    # Extract dimensions
    C_in, L_in = input.shape # [N_state, N_stack]
    C_out, _, K = weight.shape # [N_state, N_state, N_kernel]

    # Calculate the output length
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    # Initialize the output tensor
    output = np.zeros((C_out, L_out))

    # Apply padding to the input
    if padding > 0:
        input_padded = np.pad(input, ((0, 0), (padding, padding)), mode='constant')
    else:
        input_padded = input

    # Perform the convolution
    for c_out in range(C_out):
        for l_out in range(L_out):
            for c_in in range(C_in):
                for k in range(K):
                    l_in = l_out * stride + k * dilation
                    output[c_out, l_out] += input_padded[c_in, l_in] * weight[c_out, c_in, k]
        if bias is not None:
            output[c_out, :] += bias[c_out]

    return output

# test numpy implementation before mapping to c++
obs = torch.zeros(1, N_obs).to("cpu")
# buf = torch.zeros(1, N_act+N_obs, N_stack)
buf = torch.zeros(1, N_obs, N_stack)
for i in range(N_stack):
    # buf[..., i] = 0.001 * i * torch.ones(N_act+N_obs)
    buf[..., i] = 0.1 * i * torch.ones(N_obs)

task = torch.ones(1, N_task).to("cpu")
if cfg_dict["env"]["task"]["taskSet"].get("EvalSet", None):
    task = torch.tensor(
        cfg_dict["env"]["task"]["taskSet"]["EvalSet"][0], dtype=torch.float32
    ).to("cpu")
    task = task.unsqueeze(0)
task /= task.norm(p=1)

z = agent.adaptor.forward(buf)
print("torch adaptor output", z)

x = torch.cat([obs, z], dim=-1)
out = agent.policy.forward(x, task)
print("torch policy output", out)

# numpy starts here
obs = obs.squeeze().detach().numpy()
task = task.squeeze().detach().numpy()
buf = buf.squeeze().detach().numpy()

padding=N_padding
stride=1
dilation=1

cnn_w1 = np.array(cnn_w1)
cnn_b1 = np.array(cnn_b1)
if N_tcn_layers>1:
    cnn_w2 = np.array(cnn_w2)
    cnn_b2 = np.array(cnn_b2)
cnn_l_w = np.array(cnn_l_w)
cnn_l_b = np.array(cnn_l_b)

cnn_lin_w = np.array(cnn_lin_w)
cnn_lin_b = np.array(cnn_lin_b)
cnn_lout_w = np.array(cnn_lout_w)
cnn_lout_b = np.array(cnn_lout_b)

# cnn inference
x = conv1d_numpy(buf, cnn_w1, cnn_b1, padding=(N_kernel-1)*1, stride=stride, dilation=1) # [N_obs, N_stack+padding]
x = chomp1d(x, (N_kernel-1)*1) # [N_obs, N_stack]
x = np.maximum(0, x)
x = np.maximum(0, x + buf)

if N_tcn_layers>1:
    x_tmp = conv1d_numpy(x, cnn_w2, cnn_b2, padding=(N_kernel-1)*2, stride=stride, dilation=2) # [N_obs, N_stack+padding]
    x_tmp = chomp1d(x_tmp, (N_kernel-1)*2) # [N_obs, N_stack]
    x_tmp = np.maximum(0, x_tmp)
    x = np.maximum(0, x+x_tmp)

x = x[:,-1]
z = linear_layer(cnn_l_b, cnn_l_w, x, activation=False)

z = linear_layer(cnn_lin_b, cnn_lin_w, z)
z = linear_layer(cnn_lout_b, cnn_lout_w, z, activation=False)

print("numpy adaptor output", z)

obs = np.concatenate([obs, z])
w = np.concatenate([obs, task])

win_w = np.array(win_w)
win_b = np.array(win_b)

wout_w = np.array(wout_w)
wout_b = np.array(wout_b)

lin_w = np.array(lin_w)
lin_b = np.array(lin_b)

mean_w = np.array(mean_w)
# mean_b = np.array(mean_b)

# policy inference

w = linear_layer(win_b, win_w, w)
w = linear_layer(wout_b, wout_w, w, type="softmax")

x = composition_layer(lin_b, lin_w, w, obs)
# x = composition_layer(l0_b, l0_w, w, x)
out = composition_layer(None, mean_w, w, x, activation=False)

print("numpy policy output:", out)

# generate CPP

filename = "NN_Parameters.h"
print("save as", filename)
fp = open(filename, "w")

fp.write("\n")
fp.write("#pragma once\n")
fp.write("#ifndef __NN_PARAMETERS_DEF_H__\n")
fp.write("#define __NN_PARAMETERS_DEF_H__\n")
fp.write("#include <vector>\n")
fp.write("\n")

fp.write("namespace NN{\n")

fp.write("\n")
fp.write("static constexpr int N_ACT = %i;\n" % N_act)
fp.write("static constexpr int N_OBS = %i;\n" % N_obs)
fp.write("static constexpr int N_TASK = %i;\n" % N_task)
fp.write("static constexpr int N_CONTEXT = %i;\n" % N_context)
fp.write("static constexpr int N_LATENT = %i;\n" % N_latent)
fp.write("static constexpr int N_HIDDEN = %i;\n" % N_hidden)

fp.write("static constexpr int N_TCN_LAYER = %i;\n" % N_tcn_layers)
fp.write("static constexpr int N_TCN_HIDDEN = %i;\n" % N_tcn_hidden)
fp.write("static constexpr int N_KERNEL = %i;\n" % N_kernel)
fp.write("static constexpr int N_STACK = %i;\n" % N_stack)
fp.write("static constexpr int N_PADD = %i;\n" % N_padding)

fp.write("\n")
fp.write("static constexpr float AVEL_LIM = %f;\n" % avel_lim)
fp.write("static constexpr float VEL_LIM = %f;\n" % vel_lim)
fp.write("static constexpr float POS_LIM = %f;\n" % pos_lim)
fp.write("static constexpr float AUTHORITY = %f;\n" % authority)

##### NN params ####

# placeholder
fp.write("\n")
fp.write(f"std::vector<float> OBS = ")
fp.write("{\n")
for i in range(N_obs):
    fp.write("%f, " % obs[i])
fp.write("};\n")

fp.write(f"std::vector<float> TASK = ")
fp.write("{\n")
for i in range(N_task):
    fp.write("%f, " % task[i])
fp.write("};\n")

# fp.write(f"std::vector<std::vector<float>> BUFFER = ")
# fp.write("{\n")
# for i in range(N_obs+N_act):
#     fp.write("{")
#     for j in range(N_stack):
#         fp.write("%f, " % buf[i][j])
#     fp.write("},\n")
# fp.write("};\n")

# cnn_w1 = weight_norm(cnn_c1_wv, cnn_c1_wg) # [N_obs, N_obs, N_kernel]
dimension = N_obs
fp.write("\n")
fp.write(f"std::vector<std::vector<std::vector<float>>> CNN_W1 = ")
fp.write("{\n")
for i in range(dimension):
    fp.write("{\n")
    for j in range(dimension):
        fp.write("{")
        for k in range(N_kernel):
            fp.write("%f, " % cnn_w1[i][j][k])
        fp.write("},\n")
    fp.write("},\n")
fp.write("};\n")

# cnn_b1 = d['tcn.tcn.network.0.conv1.bias'] # [N_obs]
fp.write("\n") 
fp.write(f"std::vector<float> CNN_B1 = ")
# fp.write(f"const float CNN_B1[{N_obs}] = ")
fp.write("{\n")
for i in range(N_obs):
    fp.write("%f, " % cnn_b1[i])
fp.write("};\n")

if N_tcn_layers>1:
    # cnn_w2 = weight_norm(cnn_c2_wv, cnn_c2_wg)  # [N_obs, N_obs, N_kernel]
    fp.write("\n")
    fp.write(f"std::vector<std::vector<std::vector<float>>> CNN_W2 = ")
    fp.write("{\n")
    for i in range(N_obs):
        fp.write("{\n")
        for j in range(N_obs):
            fp.write("{")
            for k in range(N_kernel):
                fp.write("%f, " % cnn_w2[i][j][k])
            fp.write("},\n")
        fp.write("},\n")
    fp.write("};\n")
    
    # cnn_b2 = d['tcn.tcn.network.0.conv2.bias'] # [N_obs] 
    fp.write("\n")
    fp.write(f"std::vector<float> CNN_B2 = ")
    fp.write("{\n")
    for i in range(N_obs):
        fp.write("%f, " % cnn_b2[i])
    fp.write("};\n")


# cnn_l_w = d['tcn.linear.weight'] # [N_latent, N_obs] 
fp.write("\n")
fp.write(f"std::vector<std::vector<float>> CNN_LW = ")
fp.write("{\n")
for i in range(N_latent):
    fp.write("{")
    for j in range(N_obs):
        fp.write("%f, " % cnn_l_w[i][j])
    fp.write("},\n")
fp.write("};\n")

# cnn_l_b = d['tcn.linear.bias'] # [N_latent] 
fp.write("\n")
fp.write(f"std::vector<float> CNN_LB = ")
fp.write("{\n")
for i in range(N_latent):
    fp.write("%f, " % cnn_l_b[i])
fp.write("};\n")

# cnn_lin_w = d['fcn.lin.layer.weight'] # [N_tcn_hidden, N_latent]
fp.write("\n")
fp.write(f"std::vector<std::vector<float>> CNN_LIN_W = ")
fp.write("{\n")
for i in range(N_tcn_hidden):
    fp.write("{")
    for j in range(N_latent):
        fp.write("%f, " % cnn_lin_w[i][j])
    fp.write("},\n")
fp.write("};\n")

# cnn_lin_b = d['fcn.lin.layer.bias'] # [N_tcn_hidden]
fp.write("\n")
fp.write(f"std::vector<float> CNN_LIN_B = ")
fp.write("{\n")
for i in range(N_tcn_hidden):
    fp.write("%f, " % cnn_lin_b[i])
fp.write("};\n")

# cnn_lout_w = d['fcn.lout.layer.weight'] # [N_latent, N_tcn_hidden]
fp.write("\n")
fp.write(f"std::vector<std::vector<float>> CNN_LOUT_W = ")
fp.write("{\n")
for i in range(N_latent):
    fp.write("{")
    for j in range(N_tcn_hidden):
        fp.write("%f, " % cnn_lout_w[i][j])
    fp.write("},\n")
fp.write("};\n")

# cnn_lout_b = d['fcn.lout.layer.bias'] # [N_latent]
fp.write("\n")
fp.write(f"std::vector<float> CNN_LOUT_B = ")
fp.write("{\n")
for i in range(N_latent):
    fp.write("%f, " % cnn_lout_b[i])
fp.write("};\n")

# win_w = d['model.w_in.layer.weight'] # [N_hidden, N_obs+N_latent+N_task]
fp.write("\n")
fp.write(f"std::vector<std::vector<float>> WIN_W = ")
fp.write("{\n")
for i in range(N_hidden):
    fp.write("{")
    for j in range(N_obs+N_latent+N_task):
        fp.write("%f, " % win_w[i][j])
    fp.write("},\n")
fp.write("};\n")

# win_b = d['model.w_in.layer.bias'] # [N_hidden]
fp.write("\n")
fp.write(f"std::vector<float> WIN_B = ")
fp.write("{\n")
for i in range(N_hidden):
    fp.write("%f, " % win_b[i])
fp.write("};\n")

# wout_w = d['model.w_out.layer.weight'] # [N_context, N_hidden]
fp.write("\n")
fp.write(f"std::vector<std::vector<float>> WOUT_W = ")
fp.write("{\n")
for i in range(N_context):
    fp.write("{")
    for j in range(N_hidden):
        fp.write("%f, " % wout_w[i][j])
    fp.write("},\n")
fp.write("};\n")

# wout_b = d['model.w_out.layer.bias'] # [N_context]
fp.write("\n")
fp.write(f"std::vector<float> WOUT_B = ")
fp.write("{\n")
for i in range(N_context):
    fp.write("%f, " % wout_b[i])
fp.write("};\n")

# lin_w = d['model.l_in.weight'] # [N_context, N_hidden, N_obs+N_latent]
fp.write("\n")
fp.write(f"std::vector<std::vector<std::vector<float>>> LIN_W = ")
fp.write("{\n")
for i in range(N_context):
    fp.write("{\n")
    for j in range(N_hidden):
        fp.write("{")
        for k in range(N_obs+N_latent):
            fp.write("%f, " % lin_w[i][j][k])
        fp.write("},\n")
    fp.write("},\n")
fp.write("};\n")

# lin_b = d['model.l_in.bias'] # [N_context, N_hidden]
fp.write("\n")
fp.write(f"std::vector<std::vector<float>> LIN_B = ")
fp.write("{\n")
for i in range(N_context):
    fp.write("{")
    for j in range(N_hidden):
        fp.write("%f, " % lin_b[i][j])
    fp.write("},\n")
fp.write("};\n")

# l0_w = d['model.layers.0.weight'] # [N_context, N_hidden, N_hidden]
# fp.write("\n")
# fp.write(f"std::vector<std::vector<std::vector<float>>> L0_W = ")
# fp.write("{\n")
# for i in range(N_context):
#     fp.write("{\n")
#     for j in range(N_hidden):
#         fp.write("{")
#         for k in range(N_hidden):
#             fp.write("%f, " % l0_w[i][j][k])
#         fp.write("},\n")
#     fp.write("},\n")
# fp.write("};\n")

# l0_b = d['model.layers.0.bias'] # [N_context, N_hidden]
# fp.write("\n")
# fp.write(f"std::vector<std::vector<float>> L0_B = ")
# fp.write("{\n")
# for i in range(N_context):
#     fp.write("{")
#     for j in range(N_hidden):
#         fp.write("%f, " % l0_b[i][j])
#     fp.write("},\n")
# fp.write("};\n")

# mean_w = d['model.mean_linear.weight'] # [N_context, N_act, N_hidden]
fp.write("\n")
fp.write(f"std::vector<std::vector<std::vector<float>>> MEAN_W = ")
fp.write("{\n")
for i in range(N_context):
    fp.write("{\n")
    for j in range(N_act):
        fp.write("{")
        for k in range(N_hidden):
            fp.write("%f, " % mean_w[i][j][k])
        fp.write("},\n")
    fp.write("},\n")
fp.write("};\n")

# mean_b = d['model.mean_linear.bias'] # [N_context, N_act]
# fp.write("\n")
# fp.write(f"std::vector<std::vector<float>> MEAN_B = ")
# fp.write("{\n")
# for i in range(N_context):
#     fp.write("{")
#     for j in range(N_act):
#         fp.write("%f, " % mean_b[i][j])
#     fp.write("},\n")
# fp.write("};\n")

##### NN params ####

fp.write("\n")
fp.write("}\n")
fp.write("#endif\n")

fp.close()
