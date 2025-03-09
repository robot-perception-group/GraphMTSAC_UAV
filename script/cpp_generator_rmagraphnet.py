import json
import os
import sys

import numpy as np
import copy
import isaacgym
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.append(dir_path + "/..")
sys.path.append(dir_path + "/../common")

from common.torch_jit_utils import *
from run import get_agent

# ----------------------------------------------------------------
# 1) Load model and config
# ----------------------------------------------------------------
log_folder = "logs/"
model_folder = "rmamtsac/Quadcopter/2025-01-28-10-06-00" 
model_checkpoint = "phase2_best"

cfg_path = log_folder + model_folder + "/cfg"
model_path = model_folder + "/" + model_checkpoint + "/"

with open(cfg_path) as f:
    cfg_dict = json.load(f)

cfg_dict["env"]["num_envs"] = 1
cfg_dict["agent"]["save_model"] = False
cfg_dict["agent"]["load_model"] = True
cfg_dict["agent"]["model_path"] = model_path
cfg_dict["agent"]["phase"] = 3

agent = get_agent(cfg_dict)

print(agent.policy)
total_params = sum(p.numel() for p in agent.policy.parameters())
model_size = total_params * 4 / 1000
print("number of parameters:", total_params, ", size [KB]:", model_size)
if model_size > 70:
    print("[Warning] model size may exceed flight controller memory limits 70 [KB].")

agent.adaptor = agent.adaptor.eval().to("cpu")
agent.policy = agent.policy.to("cpu")

N_act = agent.action_dim
N_state = agent.state_dim - N_act
N_goal = agent.goal_dim
N_task = agent.feature_dim
N_latent = agent.latent_dim

pos_lim = agent.env.pos_lim
avel_lim = agent.env.avel_lim
authority = agent.env.control_authority

# parameters
N_hidden = cfg_dict["agent"]["policy_net_kwargs"]["hidden_dim"]
N_embedding = cfg_dict["agent"]["graph"]["embedding_dim"]
use_layernorm = cfg_dict["agent"]["policy_net_kwargs"]["use_layernorm"]

N_tcn_layers = cfg_dict["agent"]["adaptor_net_kwargs"]["num_tcn_layers"]
N_tcn_hidden = cfg_dict["agent"]["adaptor_net_kwargs"]["hidden_dim"]
N_kernel = cfg_dict["agent"]["adaptor_net_kwargs"]["kernel_size"]
N_padding = cfg_dict["agent"]["adaptor_net_kwargs"]["kernel_size"]-1
N_stack = agent.stack_size

# ----------------------------------------------------------------
# 2) Extract the relevant policy params from state_dict
# ----------------------------------------------------------------
# policy
d = agent.policy.state_dict()
print("policy net:", d.keys())

A              = d['model.A']                     # [N_tot, N_tot]
act_emb_w      = d['model.embeddings.0.weight']  # [N_embedding, 1]
act_emb_b      = d['model.embeddings.0.bias']    # [N_embedding]
ang_emb_w      = d['model.embeddings.1.weight']    # [N_embedding, 1]
ang_emb_b      = d['model.embeddings.1.bias']      # [N_embedding]
angvel_emb_w   = d['model.embeddings.2.weight']   # [N_embedding, 1]
angvel_emb_b   = d['model.embeddings.2.bias']      # [N_embedding]
latent_emb_w   = d['model.embeddings.3.weight']     # [N_embedding, 1]
latent_emb_b   = d['model.embeddings.3.bias']       # [N_embedding]
task_emb_w     = d['model.embeddings.4.weight']     # [N_embedding, 1]
task_emb_b     = d['model.embeddings.4.bias']       # [N_embedding]

gcn0_w   = d['model.gcn_layers.0.lin.weight']         # [out_dim, in_dim] = [N_embedding, N_hidden]
gcn0_b   = d['model.gcn_layers.0.lin.bias']           # [N_hidden]
if use_layernorm:
    gcn0_ln_w = d['model.gcn_layers.0.ln.weight']         # [N_hidden]
    gcn0_ln_b = d['model.gcn_layers.0.ln.bias']           # [N_hidden]

lout_w   = d['model.l_out.weight']                    # [A, N_out, N_in]  (ParallelFC)
lout_b   = d['model.l_out.bias']                      # [A, N_out]

mean_w   = d['model.mean_linear.weight']              # [A, 1, N_in]

# Plot adjacency matrix (Theta)
A_np = A.numpy()
adjacency = torch.clip(A,-1,1)
adjacency_np = adjacency.numpy()
        
plt.matshow(A_np)
plt.show()
plt.savefig("graph.png")
plt.close()

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

cnn_lin0_w = d['fcn.layers.0.layer.weight'] # [N_tcn_hidden, N_tcn_hidden]
cnn_lin0_b = d['fcn.layers.0.layer.bias'] # [N_tcn_hidden]

cnn_lin1_w = d['fcn.layers.1.layer.weight'] # [N_tcn_hidden, N_tcn_hidden]
cnn_lin1_b = d['fcn.layers.1.layer.bias'] # [N_tcn_hidden]

cnn_lout_w = d['fcn.lout.layer.weight'] # [N_latent, N_tcn_hidden]
cnn_lout_b = d['fcn.lout.layer.bias'] # [N_latent]

# ----------------------------------------------------------------
# 3) Some helper functions in NumPy
# ----------------------------------------------------------------
def softmax(x):
    """Compute softmax values for vector x."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def linear_layer(bias, weight, x, activation=True, type="relu"):
    """
    weight: [outdim, indim]
    bias  : [outdim]
    x     : [indim]
    returns [outdim]
    """
    out = bias + weight.dot(x)
    if activation and type == "relu":
        return np.maximum(0, out)
    elif activation and type == "softmax":
        return softmax(out)
    else:
        return out

def embed_scalars_numpy(x, embed_weight, embed_bias):
    """
    x            : np.ndarray of shape [B, X] containing scalars
    embed_weight : shape [hdim, 1] in PyTorch => we transpose for matmul => [1, hdim] in local usage
    embed_bias   : shape [hdim]

    Returns: np.ndarray of shape [B, X, hdim]
    """
    B, X = x.shape
    # We'll interpret embed_weight as [hdim, 1]
    # Flatten x => [B*X, 1]
    x_flat = x.reshape(-1, 1)  # shape [B*X, 1]

    # out_flat => [B*X, hdim] = x_flat @ embed_weight^T + bias
    # But we can do an explicit dot if we want:
    out_flat = x_flat.dot(embed_weight.T)  # => shape [B*X, hdim]
    out_flat += embed_bias  # broadcast add

    # Reshape => [B, X, hdim]
    return out_flat.reshape(B, X, -1)

def gcn_layer_numpy(
    H,               # [B, N_nodes, in_dim]
    A,               # [N_nodes, N_nodes]
    W,               # [out_dim, in_dim]
    b,               # [out_dim]
    use_layernorm=False,
    ln_weight=None,  # [out_dim]
    ln_bias=None,    # [out_dim]
    eps=1e-5
):
    """
    Replicates:
      out = ReLU( LN( Linear( A @ H ) ) ) if use_layernorm else ReLU( Linear( A @ H ) )
    Returns: [B, N_nodes, out_dim]
    """
    B, N_nodes, in_dim = H.shape
    out_dim, _ = W.shape

    # 1) out_matmul = A @ H per batch
    out_matmul = np.zeros((B, N_nodes, in_dim), dtype=H.dtype)
    for i in range(B):
        # shape => [N_nodes, in_dim]
        out_matmul[i] = A @ H[i]

    # 2) Linear: out_lin = out_matmul @ W^T + b
    out_lin = np.zeros((B, N_nodes, out_dim), dtype=H.dtype)
    for i in range(B):
        tmp = out_matmul[i].dot(W.T)
        tmp += b  # broadcast over axis=0
        out_lin[i] = tmp

    # 3) Optional LayerNorm over last dimension
    if use_layernorm and ln_weight is not None and ln_bias is not None:
        for i in range(B):
            for n in range(N_nodes):
                row = out_lin[i, n]  # shape [out_dim]
                mean = np.mean(row)
                var = np.var(row)
                row_norm = (row - mean) / np.sqrt(var + eps)
                row_norm = row_norm * ln_weight + ln_bias
                out_lin[i, n] = row_norm

    # 4) ReLU
    out_relu = np.maximum(out_lin, 0)
    return out_relu

def parallel_fc_numpy(
    inputs,      # shape [B, n_parallels, in_features]
    weight,      # shape [n_parallels, out_features, in_features]
    bias=None,        # shape [n_parallels, out_features]
    activation=None,
    use_bias=True,
):
    """
    Replicates ParallelFC.forward in a simplified manner (no group norm by default).
    Returns: [B, n_parallels, out_features]
    """
    B, n_parallels, in_features = inputs.shape
    _, out_features, _ = weight.shape

    # Prepare output [B, n_parallels, out_features]
    out = np.zeros((B, n_parallels, out_features), dtype=inputs.dtype)

    # 1) For each parallel i, do (x_i @ W_i^T + b_i)
    for i in range(n_parallels):
        # weight[i]: [out_features, in_features]
        # inputs[:, i, :]: [B, in_features]
        # => matmul => [B, out_features]
        if use_bias:
            out[:, i, :] = inputs[:, i, :].dot(weight[i].T) + bias[i]
        else:
            out[:, i, :] = inputs[:, i, :].dot(weight[i].T)

    # 2) If you need groupnorm or layernorm “per parallel,” handle here.
    #    In your PyTorch code, it’s a GroupNorm(n_parallels, n_parallels*out_features).
    #    That is a bit more involved in NumPy. For simplicity, we skip it here.
    #    ...
    
    # 3) Activation
    if activation == "relu":
        out = np.maximum(out, 0)

    return out

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

# ----------------------------------------------------------------
# 4) Quick Test: Torch vs NumPy for one forward pass
# ----------------------------------------------------------------

# We'll do a very simplified test: zero obs => pass to policy => compare.
obs = torch.zeros(1, N_state + N_act + N_goal).to("cpu")  # [1, S+A+G]
buf = torch.zeros(1, N_state + N_act, N_stack)
for i in range(N_stack):
    # buf[..., i] = 0.001 * i * torch.ones(N_act+N_obs)
    buf[..., i] = 0.1 * i * torch.ones(N_state + N_act)

task = torch.ones(1, N_task).to("cpu")

# Possibly load real eval set
if cfg_dict["env"]["task"]["taskSet"].get("EvalSet", None):
    task = torch.tensor(
        cfg_dict["env"]["task"]["taskSet"]["EvalSet"][0], dtype=torch.float32
    ).unsqueeze(0).to("cpu")
task /= task.norm(p=1)

# Torch forward
z = agent.adaptor.forward(buf)
print("Torch adaptor output", z)

x = torch.cat([obs, z], dim=-1)
torch_out = agent.policy.forward(x, task)
print("Torch policy output:", torch_out)


# Now replicate the *first part* of the forward in NumPy
# forward adaptor
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
cnn_lin0_w = np.array(cnn_lin0_w)
cnn_lin0_b = np.array(cnn_lin0_b)
cnn_lin1_w = np.array(cnn_lin1_w)
cnn_lin1_b = np.array(cnn_lin1_b)
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
z = linear_layer(cnn_lin0_b, cnn_lin0_w, z)
z = linear_layer(cnn_lin1_b, cnn_lin1_w, z)
z = linear_layer(cnn_lout_b, cnn_lout_w, z, activation=False)

print("numpy adaptor output", z)

# forward policy
rb_ang    = obs[:, 0:3]
rb_angvel = obs[:, 3:6]
prev_act  = obs[:, 6:9]
goal      = obs[:, 9:12]

rb_ang_np    = rb_ang.squeeze(0).detach().numpy()    # shape [3,]
rb_angvel_np = rb_angvel.squeeze(0).detach().numpy() # shape [3,]
prev_act_np  = prev_act.squeeze(0).detach().numpy()  # shape [3,]
goal_np      = goal.squeeze(0).detach().numpy()      # [N_goal,]
latent_np    = z   # [N_latent,]
task_np      = task.squeeze(0).detach().numpy()      # shape [N_task,]

# 4.1) Embeddings
H_action   = embed_scalars_numpy(prev_act_np.reshape(1, -1), act_emb_w.numpy(),    act_emb_b.numpy())   # => [1, A, hdim]
H_ang    = embed_scalars_numpy(rb_ang_np.reshape(1, -1),     ang_emb_w.numpy(),    ang_emb_b.numpy())    # => [B=1, X=3, hdim]
H_angvel = embed_scalars_numpy(rb_angvel_np.reshape(1, -1),  angvel_emb_w.numpy(), angvel_emb_b.numpy()) # => [1, 3, hdim]
H_goal   = embed_scalars_numpy(goal_np.reshape(1, -1),       ang_emb_w.numpy(),    ang_emb_b.numpy())    # => [1, 3, hdim] (same embed as angle)
H_latent = embed_scalars_numpy(latent_np.reshape(1, -1),     latent_emb_w.numpy(), latent_emb_b.numpy())    # => [1, 3, hdim] (same embed as angle)
H_task   = embed_scalars_numpy(task_np.reshape(1, -1),       task_emb_w.numpy(),   task_emb_b.numpy())   # => [1, T, hdim]

# 4.2) Concatenate node embeddings + action init
# For your code, total obs nodes = 3 + 3 + 3 + 3 + T = 12 + T
H_concat = np.concatenate([H_action, H_ang, H_angvel, H_goal, H_latent, H_task], axis=1)  
# shape => [B, N_obs_nodes + A, hdim]

# 4.3) Apply a single GCN layer (like gcn_layers.0)
if use_layernorm:
    H_gcn0 = gcn_layer_numpy(
        H_concat,
        adjacency_np,
        gcn0_w.numpy(), gcn0_b.numpy(),
        use_layernorm=use_layernorm,
        ln_weight=gcn0_ln_w.numpy(),
        ln_bias=gcn0_ln_b.numpy(),
    )
else:
    H_gcn0 = gcn_layer_numpy(
        H_concat,
        adjacency_np,
        gcn0_w.numpy(), gcn0_b.numpy(),
        use_layernorm=False
    )

# (If you have multiple GCN layers, you'd repeat for layer1, layer2, etc.)

# ----------------------------------------------------------------
# 5) Compare partial results or continue to final action
# ----------------------------------------------------------------
# The next steps in Torch policy are:
#   -> parallelFC (l_out)
#   -> mean_linear, log_std_linear
#
# - Slice out action nodes from the GCN output
num_obs_nodes = N_state + N_goal + N_latent + N_task  
num_action_nodes = N_act
total_nodes = num_obs_nodes + num_action_nodes
H_actions = H_gcn0[:, : num_action_nodes, :]
# shape => [B, action_dim, hidden_dim]

# - l_out (ParallelFC)
x_np = parallel_fc_numpy(
    inputs=H_actions,
    weight=lout_w.numpy(),  # [A, hidden_dim, hidden_dim]
    bias=lout_b.numpy(),    # [A, hidden_dim]
    activation="relu"
)
# shape => [B, A, hidden_dim]

# 3) mean_linear => shape [B, A, 1], then squeeze => [B, A]
mean_out_np = parallel_fc_numpy(
    inputs=x_np,
    weight=mean_w.numpy(),  # [A, 1, hidden_dim]
    use_bias=False,
    activation=None
)
mean_out_np = mean_out_np.squeeze(-1)  # => shape [B, A]

print("Final mean from NumPy:", mean_out_np.shape, "\n", mean_out_np)

# ----------------------------------------------------------------
# 6) Generate Cpp code
# ----------------------------------------------------------------

def write_flattened_array(fp, var_name, arr_2d):
    """
    Writes a flattened 2D float array (shape: [dim1, dim2]) to the header as a 1D array.
    """
    fp.write(f"static const std::vector<float> {var_name} = {{\n")
    for row in arr_2d:
        for val in row:
            fp.write(f"    {val}f, ")  # Add commas after each value
        fp.write("\n")  # Line break for readability
    fp.write("};\n\n")

def write_flattened_3d_array(fp, var_name, arr_3d):
    """
    Writes a flattened 3D float array (shape: [dim1, dim2, dim3]) to the header as a 1D array.
    """
    fp.write(f"static const std::vector<float> {var_name} = {{\n")
    for matrix in arr_3d:       # Iterate over the first dimension
        for row in matrix:      # Iterate over the second dimension
            for val in row:     # Iterate over the third dimension
                fp.write(f"    {val}f, ")  # Add value with 'f' and a comma
            fp.write("\n")  # Line break for readability
    fp.write("};\n\n")

def write_1d_array(fp, var_name, arr_1d):
    """
    Writes a 1D float array (shape: [dim]) to the header, e.g.:
      static const std::vector<float> VAR = {
          0.1f, 0.2f, ...
      };
    """
    fp.write(f"static const std::vector<float> {var_name} = {{\n")
    for val in arr_1d:
        fp.write(f"    {val}f,\n")
    fp.write("};\n\n")

def write_2d_array(fp, var_name, arr_2d):
    """
    Writes a 2D float array (shape: [dim1, dim2]) to the header, e.g.:
      static const std::vector<std::vector<float>> VAR = {
          {0.1f, 0.2f},
          {0.3f, 0.4f}
      };
    """
    fp.write(f"static const std::vector<std::vector<float>> {var_name} = {{\n")
    for row in arr_2d:
        fp.write("    { ")
        fp.write(", ".join(f"{val}f" for val in row))
        fp.write(" },\n")
    fp.write("};\n\n")

def write_3d_array(fp, var_name, arr_3d):
    """
    Writes a 3D float array (shape: [dim1, dim2, dim3]) to the header, e.g.:
      static const std::vector<std::vector<std::vector<float>>> VAR = {
          {
            {1.0f, 2.0f},
            {3.0f, 4.0f}
          },
          {
            {5.0f, 6.0f},
            {7.0f, 8.0f}
          }
      };
    """
    fp.write(f"static const std::vector<std::vector<std::vector<float>>> {var_name} = {{\n")
    for mat in arr_3d:
        fp.write("    {\n")
        for row in mat:
            fp.write("        { ")
            fp.write(", ".join(f"{val}f" for val in row))
            fp.write(" },\n")
        fp.write("    },\n")
    fp.write("};\n\n")


def export_params_to_header(d_policy, d_adaptor, filename="NN_Parameters.h"):
    """
    Writes the relevant model parameters to a C++ header file.
    The dictionary `d_policy` is assumed to have the keys:
        'model.A'                   => shape [N_tot, N_tot]
        'model.angle_embedding.weight'  => shape [N_embedding, 1]
        'model.angle_embedding.bias'    => shape [N_embedding]
        'model.angvel_embedding.weight' => shape [N_embedding, 1]
        'model.angvel_embedding.bias'   => shape [N_embedding]
        'model.latent_embedding.weight' => shape [N_embedding, 1]
        'model.latent_embedding.bias'   => shape [N_embedding]
        'model.task_embedding.weight'   => shape [N_embedding, 1]
        'model.task_embedding.bias'     => shape [N_embedding]

        'model.gcn_layers.0.lin.weight' => shape [N_embedding, N_hidden]
        'model.gcn_layers.0.lin.bias'   => shape [N_hidden]
        'model.gcn_layers.0.ln.weight'  => shape [N_hidden]
        'model.gcn_layers.0.ln.bias'    => shape [N_hidden]

        'model.l_out.weight'            => shape [A, N_out, N_in]
        'model.l_out.bias'              => shape [A, N_out]
        'model.l_out._ln.weight'        => shape [A*N_out]
        'model.l_out._ln.bias'          => shape [A*N_out]

        'model.mean_linear.weight'      => shape [A, 1, N_in]
    """
    print("Saving as:", filename)
    
    # Convert all torch tensors to NumPy arrays (and float) for safety
    # Example:  arr = d["model.Theta"].cpu().numpy().astype(float)
    # We'll define a small helper to do that:
    def to_np(d, key):
        return d[key].cpu().numpy().astype(float)

    # Retrieve arrays
    # A_np          = to_np('model.A')                  # [N_tot, N_tot]
    act_emb_w_np      = to_np(d_policy, 'model.embeddings.0.weight')   # [N_hidden, 1]
    act_emb_b_np      = to_np(d_policy, 'model.embeddings.0.bias')     # [N_hidden]

    ang_emb_w_np      = to_np(d_policy, 'model.embeddings.1.weight')   # [N_hidden, 1]
    ang_emb_b_np      = to_np(d_policy, 'model.embeddings.1.bias')     # [N_hidden]
    
    angvel_emb_w_np   = to_np(d_policy, 'model.embeddings.2.weight')  # [N_hidden, 1]
    angvel_emb_b_np   = to_np(d_policy, 'model.embeddings.2.bias')    # [N_hidden]
    
    latent_emb_w_np   = to_np(d_policy, 'model.embeddings.3.weight')  # [N_hidden, 1]
    latent_emb_b_np   = to_np(d_policy, 'model.embeddings.3.bias')    # [N_hidden]

    task_emb_w_np     = to_np(d_policy, 'model.embeddings.4.weight')    # [N_hidden, 1]
    task_emb_b_np     = to_np(d_policy, 'model.embeddings.4.bias')      # [N_hidden]

    gcn0_w_np         = to_np(d_policy, 'model.gcn_layers.0.lin.weight')  # [N_hidden, N_hidden]
    gcn0_b_np         = to_np(d_policy, 'model.gcn_layers.0.lin.bias')    # [N_hidden]
    if use_layernorm:
        gcn0_ln_w_np      = to_np(d_policy, 'model.gcn_layers.0.ln.weight')   # [N_hidden]
        gcn0_ln_b_np      = to_np(d_policy, 'model.gcn_layers.0.ln.bias')     # [N_hidden]

    lout_w_np         = to_np(d_policy, 'model.l_out.weight')             # [A, N_out, N_in]
    lout_b_np         = to_np(d_policy, 'model.l_out.bias')               # [A, N_out]

    mean_w_np         = to_np(d_policy, 'model.mean_linear.weight')       # [A, 1, N_in]

    """
    Writes the relevant model parameters to a C++ header file.
    The dictionary `d_adaptor` is assumed to have the keys:
        'tcn.tcn.network.0.conv1.bias'  => shape [N_obs]
        'tcn.tcn.network.0.conv1.weight_g' => shape [N_obs, 1, 1]
        'tcn.tcn.network.0.conv1.weight_v' => shape [N_obs, N_obs, N_kernel]
        'tcn.linear.weight'              => shape [N_latent, N_obs]
        'tcn.linear.bias'                => shape [N_latent]
        'fcn.lin.layer.weight'          => shape [N_tcn_hidden, N_latent]
        'fcn.lin.layer.bias'            => shape [N_tcn_hidden]
        'fcn.layers.0.layer.weight'     => shape [N_tcn_hidden, N_tcn_hidden]
        'fcn.layers.0.layer.bias'       => shape [N_tcn_hidden]
        'fcn.layers.1.layer.weight'     => shape [N_tcn_hidden, N_tcn_hidden]
        'fcn.layers.1.layer.bias'       => shape [N_tcn_hidden]
        'fcn.lout.layer.weight'         => shape [N_latent, N_tcn_hidden]
        'fcn.lout.layer.bias'           => shape [N_latent]
    """
    cnn_b1 = to_np(d_adaptor, 'tcn.tcn.network.0.conv1.bias')  # [N_obs]
    cnn_c1_wg = d_adaptor['tcn.tcn.network.0.conv1.weight_g'] # [N_obs, 1, 1]
    cnn_c1_wv = d_adaptor['tcn.tcn.network.0.conv1.weight_v'] # [N_obs, N_obs, N_kernel]
    cnn_w1 = weight_norm(cnn_c1_wv, cnn_c1_wg) # [N_obs, N_obs, N_kernel]
    cnn_w1 = cnn_w1.cpu().numpy().astype(float)

    # if N_tcn_layers>1:
    #     cnn_b2 = d['tcn.tcn.network.1.conv1.bias'] # [N_obs]
    #     cnn_c2_wg = d['tcn.tcn.network.1.conv1.weight_g'] # [N_obs, 1, 1]
    #     cnn_c2_wv = d['tcn.tcn.network.1.conv1.weight_v'] # [N_obs, N_obs, N_kernel]
    #     cnn_w2 = weight_norm(cnn_c2_wv, cnn_c2_wg)  

    cnn_l_w = to_np(d_adaptor, 'tcn.linear.weight')  # [N_latent, N_obs]
    cnn_l_b = to_np(d_adaptor, 'tcn.linear.bias')  # [N_latent]

    cnn_lin_w = to_np(d_adaptor, 'fcn.lin.layer.weight')  # [N_tcn_hidden, N_latent]
    cnn_lin_b = to_np(d_adaptor, 'fcn.lin.layer.bias')  # [N_tcn_hidden]

    cnn_lin0_w = to_np(d_adaptor, 'fcn.layers.0.layer.weight') # [N_tcn_hidden, N_tcn_hidden]
    cnn_lin0_b = to_np(d_adaptor, 'fcn.layers.0.layer.bias') # [N_tcn_hidden]

    cnn_lin1_w = to_np(d_adaptor,'fcn.layers.1.layer.weight') # [N_tcn_hidden, N_tcn_hidden]
    cnn_lin1_b = to_np(d_adaptor,'fcn.layers.1.layer.bias') # [N_tcn_hidden]

    cnn_lout_w = to_np(d_adaptor,'fcn.lout.layer.weight') # [N_latent, N_tcn_hidden]
    cnn_lout_b = to_np(d_adaptor,'fcn.lout.layer.bias') # [N_latent]

    ################################################

    with open(filename, "w") as fp:
        fp.write("#pragma once\n")
        fp.write("#ifndef __NN_PARAMETERS_DEF_H__\n")
        fp.write("#define __NN_PARAMETERS_DEF_H__\n\n")
        fp.write("#include <vector>\n\n")
        fp.write("namespace NN {\n\n")

        #
        # write parameters
        #
        fp.write("\n")
        fp.write("static constexpr bool USE_LAYERNORM = %i;\n" % use_layernorm)

        fp.write("static constexpr int N_ACT = %i;\n" % N_act)
        fp.write("static constexpr int N_STATE = %i;\n" % N_state)
        fp.write("static constexpr int N_GOAL = %i;\n" % N_goal)
        fp.write("static constexpr int N_LATENT = %i;\n" % N_latent)
        fp.write("static constexpr int N_TASK = %i;\n" % N_task)

        fp.write("static constexpr int N_HIDDEN = %i;\n" % N_hidden)
        fp.write("static constexpr int N_EMBEDDING = %i;\n" % N_embedding)

        fp.write("static constexpr int N_TCN_LAYER = %i;\n" % N_tcn_layers)
        fp.write("static constexpr int N_TCN_HIDDEN = %i;\n" % N_tcn_hidden)
        fp.write("static constexpr int N_KERNEL = %i;\n" % N_kernel)
        fp.write("static constexpr int N_STACK = %i;\n" % N_stack)
        fp.write("static constexpr int N_PADD = %i;\n" % N_padding)

        fp.write("\n")
        fp.write("static constexpr float AVEL_LIM = %f;\n" % avel_lim)
        fp.write("static constexpr float POS_LIM = %f;\n" % pos_lim)
        fp.write("static constexpr float AUTHORITY = %f;\n" % authority)

        fp.write("\n")
        fp.write(f"std::vector<float> OBS = ")
        fp.write("{\n")
        obs_np = obs.squeeze(0).detach().numpy()
        for i in range(N_state+N_act+N_goal):
            fp.write("%f, " % obs_np[i])
        fp.write("};\n")

        fp.write(f"std::vector<float> TASK = ")
        fp.write("{\n")
        task_np = task.squeeze(0).detach().numpy()
        for i in range(N_task):
            fp.write("%f, " % task_np[i])
        fp.write("};\n")


        # fp.write(f"std::vector<std::vector<float>> BUFFER = ")
        # fp.write("{\n")
        # for i in range(N_state+N_act):
        #     fp.write("{")
        #     for j in range(N_stack):
        #         fp.write("%f, " % buf[i][j])
        #     fp.write("},\n")
        # fp.write("};\n")

        #
        # Write model parameter with the right dimensionality
        #

        # 1) Extract the first three rows
        first_rows = adjacency_np[:num_action_nodes, :]  # shape [3, N_nodes]

        # 2) Flatten row by row (C-style) => shape [3*N_nodes]
        first_rows_flat = first_rows.flatten()  # or first_rows.reshape(-1)

        # Write the extracted rows to the header file
        write_1d_array(fp, "A", first_rows_flat)          # Already 1D, write directly

        # 2) Action init (2D)
        write_flattened_array(fp, "ACT_EMB_W", act_emb_w_np)   # Flatten [N_hidden, 1] to [N_hidden]
        write_1d_array(fp, "ACT_EMB_B", act_emb_b_np)          # Already 1D, write directly

        # Save embedding weights and biases as 1D arrays (flattened)
        write_flattened_array(fp, "ANG_EMB_W", ang_emb_w_np)   # Flatten [N_hidden, 1] to [N_hidden]
        write_1d_array(fp, "ANG_EMB_B", ang_emb_b_np)          # Already 1D, write directly

        write_flattened_array(fp, "ANGVEL_EMB_W", angvel_emb_w_np)  # Flatten [N_hidden, 1] to [N_hidden]
        write_1d_array(fp, "ANGVEL_EMB_B", angvel_emb_b_np)         # Already 1D, write directly

        write_flattened_array(fp, "LATENT_EMB_W", latent_emb_w_np)  # Flatten [N_hidden, 1] to [N_hidden]
        write_1d_array(fp, "LATENT_EMB_B", latent_emb_b_np)         # Already 1D, write directly

        write_flattened_array(fp, "TASK_EMB_W", task_emb_w_np)      # Flatten [N_hidden, 1] to [N_hidden]
        write_1d_array(fp, "TASK_EMB_B", task_emb_b_np)             # Already 1D, write directly

        # 7) GCN0: weight (2D), bias (1D), LN weight (1D), LN bias (1D)
        write_flattened_array(fp, "GCN0_W", gcn0_w_np)  # Flatten and save as 1D array
        write_1d_array(fp, "GCN0_B", gcn0_b_np)
        if use_layernorm:
            write_1d_array(fp, "GCN0_LN_W", gcn0_ln_w_np)
            write_1d_array(fp, "GCN0_LN_B", gcn0_ln_b_np)
        else:
            write_1d_array(fp, "GCN0_LN_W", [])
            write_1d_array(fp, "GCN0_LN_B", [])

        # 8) l_out: 3D weight, 2D bias, LN weight/bias (1D each)
        write_flattened_3d_array(fp, "LOUT_W", lout_w_np)  # shape [A, N_out, N_in] Save as flattened array
        write_flattened_array(fp, "LOUT_B", lout_b_np)   # shape [A, N_out]

        # 9) mean_linear: 3D weight [A,1,N_in], 2D bias [A,1]
        write_flattened_3d_array(fp, "MEAN_W", mean_w_np)

        # adaptor
        write_3d_array(fp, "CNN_W1", cnn_w1)
        write_1d_array(fp, "CNN_B1", cnn_b1)
        
        write_2d_array(fp, "CNN_LW", cnn_l_w)
        write_1d_array(fp, "CNN_LB", cnn_l_b)
        
        write_2d_array(fp, "CNN_LIN_W", cnn_lin_w)
        write_1d_array(fp, "CNN_LIN_B", cnn_lin_b)
        
        write_2d_array(fp, "CNN_LIN0_W", cnn_lin0_w)
        write_1d_array(fp, "CNN_LIN0_B", cnn_lin0_b)

        write_2d_array(fp, "CNN_LIN1_W", cnn_lin1_w)
        write_1d_array(fp, "CNN_LIN1_B", cnn_lin1_b)

        write_2d_array(fp, "CNN_LOUT_W", cnn_lout_w)
        write_1d_array(fp, "CNN_LOUT_B", cnn_lout_b)

        fp.write("} // namespace NN\n\n")
        fp.write("#endif // __NN_PARAMETERS_DEF_H__\n")
        fp.close()    

# Call the function with small model data
d_policy = agent.policy.state_dict()
d_adaptor = agent.adaptor.state_dict()
export_params_to_header(d_policy, d_adaptor, filename="NN_Parameters.h")
print("Done. Check NN_Parameters.h for output.")

