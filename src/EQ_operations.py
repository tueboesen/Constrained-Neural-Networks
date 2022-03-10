import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.util.jit import compile_mode
from torch_scatter import scatter


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    """
    Computes whether a path between irreducible representations exists
    """
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

@compile_mode('script')
class Convolution(torch.nn.Module):
    r"""equivariant convolution
    Parameters
    ----------
    irreps_node_input : `Irreps`
        representation of the input node features
    irreps_node_attr : `Irreps`
        representation of the node attributes
    irreps_edge_attr : `Irreps`
        representation of the edge attributes
    irreps_node_output : `Irreps` or None
        representation of the output node features
    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)

        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel],
            torch.nn.functional.silu
        )
        self.tp = tp

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)
        self.lin3 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars,num_neighbors) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)
        node_features = self.lin1(node_input, node_attr)

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0, dim_size=node_input.shape[0]).div(num_neighbors**0.5)

        node_conv_out = self.lin2(node_features, node_attr)
        node_angle = 0.1 * self.lin3(node_features, node_attr)
        #            ^^^------ start small, favor self-connection

        cos, sin = node_angle.cos(), node_angle.sin()
        m = self.sc.output_mask
        sin = (1 - m) + sin * m
        return cos * node_self_connection + sin * node_conv_out


class Identity(torch.nn.Module):
    """
    An identity class that can be inserted and does nothing.
    """
    def __init__(self):
        super().__init__()
        return
    def forward(self, input):
        return input

class SelfInteraction(torch.nn.Module):
    """
    A self-interaction operation which mixes the node information internally among the different tensor representations.
    """
    def __init__(self, irreps_in,irreps_out):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.tp = o3.FullyConnectedTensorProduct(irreps_in,irreps_in,irreps_out)

        nd = irreps_out.dim
        nr = irreps_out.num_irreps
        degen = torch.empty(nr, dtype=torch.int64)
        m_degen = torch.empty(nr, dtype=torch.bool)
        idx = 0
        for mul, ir in irreps_out:
            li = 2 * ir.l + 1
            for i in range(mul):
                degen[idx + i] = li
                m_degen[idx + i] = ir.l == 0
            idx += i + 1
        M = m_degen.repeat_interleave(degen)
        self.register_buffer("m_scalar", M)
        return

    def forward(self, x, normalize_variance=True, eps=1e-9, debug=False):
        y = self.tp(x,x)
        if normalize_variance:
            nb, _ = y.shape
            ms = self.m_scalar
            std = torch.std(y[:, ms], dim=1)
            y[:, ms] /= std[:,None] + eps

            mv = ~ms
            if torch.sum(mv) > 0:
                tmp = y[:,mv].clone()
                yy = tmp.view(nb,-1,3).clone()
                norm1 = torch.sqrt(torch.sum(yy**2,dim=2)+eps)
                std = norm1.std(dim=1)
                yy2 = yy / (std[:,None,None]+eps)
                y[:,mv] = yy2.view(nb,-1)
        if debug:
            if normalize_variance and torch.sum(mv) > 0:
                with torch.no_grad():
                    scalars_in = torch.cat([x[:, ms].view(-1),norm1.view(-1)])
                    tmp1 = y[:, mv]
                    tmp2 = tmp1.view(nb, -1, 3)
                    norm2 = tmp2.norm(dim=2)
                    scalars_out = torch.cat([y[:, ms].view(-1),norm2.view(-1)])
                print(f"input var: {scalars_in.var():2.2f}, output var: {scalars_out.var():2.2f}")
            else:
                print(f"Normalize variance={normalize_variance}, input var: {x.var():2.2f}, output var: {y.var():2.2f}")
        return y



if __name__ == '__main__':
    pass