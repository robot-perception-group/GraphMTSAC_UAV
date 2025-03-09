import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, LinearFC)
        or isinstance(m, ParallelFC)
        or isinstance(m, CompositionalFC)
    ):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if hasattr(m, 'bias'):
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, GCNLayer):
        torch.nn.init.xavier_uniform_(m.lin.weight, gain=1)
        if hasattr(m.lin, 'bias'):
            torch.nn.init.constant_(m.lin.bias, 0)


class LinearFC(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 activation=None,
                 use_layernorm=False,
                 ):

        super().__init__()
        self._activation = activation
        self.layer = nn.Linear(input_size, output_size)

        if use_layernorm:
            self._ln = nn.LayerNorm(output_size)
        else:
            self._ln = None

    def forward(self, inputs):
        y = self.layer(inputs)

        if self._ln is not None:
            y = self._ln(y)
        
        return self._activation(y) if self._activation is not None else y
    
    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

class ParallelFC(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_parallels,
                 activation=None,
                 use_bias=True,
                 use_layernorm=False,
                 ):

        super().__init__()
        self._activation = activation
        self.use_bias = use_bias

        self.weight = nn.Parameter(torch.Tensor(n_parallels, output_size, input_size))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_parallels, output_size))

        if use_layernorm:
            self._ln = nn.GroupNorm(n_parallels, n_parallels * output_size)
        else:
            self._ln = None

    def forward(self, inputs):
        """Forward

        Args:
            inputs (torch.Tensor): with shape ``[B, n, input_size]`` or ``[B, input_size]``
        Returns:
            torch.Tensor with shape ``[B, n, output_size]``
        """
        n, k, l = self.weight.shape # n_parallels, output_size, input_size
        if inputs.ndim == 2:
            assert inputs.shape[1] == l, (
                "inputs has wrong shape %s. Expecting (B, %d)" % (inputs.shape,
                                                                  l))
            inputs = inputs.unsqueeze(0).expand(n, *inputs.shape)
        elif inputs.ndim == 3:
            assert (inputs.shape[1] == n and inputs.shape[2] == l), (
                "inputs has wrong shape %s. Expecting (B, %d, %d)" %
                (inputs.shape, n, l))
            inputs = inputs.transpose(0, 1)  # [n, B, l]
        else:
            raise ValueError("Wrong inputs.ndim=%d" % inputs.ndim)
        
        if self.use_bias:
            y = torch.baddbmm(
                self.bias.unsqueeze(1), inputs,
                self.weight.transpose(1, 2))  # [n, B, k]
        else:
            y = torch.bmm(inputs, self.weight.transpose(1, 2))

        y = y.transpose(0, 1)  # [B, n, k]
        if self._ln is not None:
            self._ln.bias.data.zero_()
            y1 = y.reshape(-1, n * k)
            y = self._ln(y1)
            y = y1.view(-1, n, k)
        
        return self._activation(y) if self._activation is not None else y


class CompositionalFC(nn.Module):
    """The code is modified from PaCo
    Lingfeng Sun, 2022, PaCo: Parameter-Compositional Multi-Task Reinforcement Learning
    """

    def __init__(self,
                 input_size,
                 output_size,
                 n_parallels,
                 activation=None,
                 use_bias=True,
                 use_layernorm=False,
                 use_comp_layer=False,
                 ):
        """
        It maintains a set of ``n`` FC parameters for learning. During forward
        computation, it composes the set of parameters using weighted average
        with the compositional weight provided as input and then performs the
        FC computation, which is equivalent to combine the pre-activation output
        from each of the ``n`` FC layers using the compositional weight, and
        then apply normalization and activation.

        Args:
            input_size (int): input size
            output_size (int): output size
            n (int): the size of the paramster set
            activation (torch.nn.functional):
            use_comp_layer (bool): apply linear layer for the compositional weight
            use_layernorm (bool): whether use layer normalization
        """
        super().__init__()
        self._activation = activation
        self.weight = nn.Parameter(torch.Tensor(n_parallels, output_size, input_size))

        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(n_parallels, output_size))

        if use_comp_layer:
            self.comp_layer = LinearFC(n_parallels, n_parallels, F.softmax)
        else:
            self.comp_layer = None

        self._n_parallels = n_parallels

        if use_layernorm:
            self._ln = nn.LayerNorm(output_size)
        else:
            self._ln = None


    def forward(self, inputs):
        """Forward

        Args:
            inputs (torch.Tensor|tuple): If a Tensor, its shape should be
            ``[B, input_size]``. If a tuple, it should contain two elements.
            The first is a Tensor with the shape of ``[B, input_size]``, the
            second is a compositional weight Tensor with the shape of ``[B, n]``
            or None. If the compositional weight is not specified (i.e. when
            inputs is not a tuple) or None, a uniform weight of one wil be used.
        Returns:
            torch.Tensor representing the final activation with shape
            ``[B, output_size]`` 
        """

        if type(inputs) == tuple:
            inputs, comp_weight = inputs
        else:
            comp_weight = None

        n, k, l = self.weight.shape # n_parallels, output_size, input_size

        if inputs.ndim == 2:
            assert inputs.shape[1] == l, (
                "inputs has wrong shape %s. Expecting (B, %d)" % (inputs.shape,
                                                                  l))
            inputs = inputs.unsqueeze(0).expand(n, *inputs.shape)

        else:
            raise ValueError("Wrong inputs.ndim=%d" % inputs.ndim)

        if self.use_bias:
            y = torch.baddbmm(
                self.bias.unsqueeze(1), inputs,
                self.weight.transpose(1, 2))  # [n, B, k]
        else:
            y = torch.bmm(inputs, self.weight.transpose(1, 2))

        y = y.transpose(0, 1)  # [B, n, k]

        if comp_weight is not None:
            assert comp_weight.ndim == 2, (
                "Wrong comp_weight.ndim=%d" % comp_weight.ndim)

            if self.comp_layer is not None:
                comp_weight = self.comp_layer(comp_weight)

            # [B, 1, n] x [B, n, k] -> [B, 1, k] -> [B, k]
            y = torch.bmm(comp_weight.unsqueeze(1), y).squeeze(1)

        else:
            y = y.sum(dim=1)

        if self._ln is not None:
            y = self._ln(y)

        if self._activation is not None:
            y = self._activation(y)

        return y


class SimpleAttention(nn.Module):
    """Simple Attention Module."""

    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        """Simple attention computation based on the inputs.
        Args:
            query (Q): shape [B, head, M, d]
            key   (K): shape [B, head, N, d]
            value (V): shape [B, head, N, d]
            where B denotes the batch size, head denotes the number of heads,
            N the number of entities, and d the feature dimension.
        Return:
            - the attended results computed as: softmax(QK^T/sqrt(d))V,
                with the shape [B, head, M, d]
            - the attention weight, with the shape [B, head, M, N]
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k))

        # [B, head, M, N]
        attention_weight = F.softmax(scores, dim=-1)

        # [B, head, M, d]
        output = torch.matmul(attention_weight, value)

        return output, attention_weight


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_layernorm=False):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(out_dim)
        
    def forward(self, H, A):
        # H: [B, N_nodes, in_dim]
        # A: [N_nodes, N_nodes]
        # Message passing: H' = ReLU(A * H * W)
        # Apply linear
        out = torch.matmul(A, H)  # [B, N_nodes, in_dim]
        out = self.lin(out)       # [B, N_nodes, out_dim]
        if self.use_layernorm:
            out = self.ln(out)
        out = F.relu(out)
        return out