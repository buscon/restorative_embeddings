"""Minimal stand-in for the few DGL calls SoundAQnet makes, so the model runs without DGL.

SoundAQnet (SoundSCaper repo) builds one complete directed graph with 8 nodes (64 edges,
including self-loops; edge k = i * 8 + j goes from node i to node j) and runs one Gated GCN
layer on a batch of such graphs. DGL has no wheels for current PyTorch/CUDA (RTX 50 series),
so the handful of operations used are re-implemented here with plain tensors:

    dgl.batch, Graph.ndata / edata, Graph.apply_edges(fn.u_add_v), Graph.update_all(fn.u_mul_e
    or fn.copy_e, fn.sum), Graph.to

install() registers this module as `dgl` and `dgl.function` in sys.modules; call it before
importing the SoundSCaper model code.
"""
import sys
import types

import torch


class _Spec:
    def __init__(self, kind, *names):
        self.kind, self.names = kind, names


def u_add_v(lhs, rhs, out):
    return _Spec("u_add_v", lhs, rhs, out)


def u_mul_e(lhs, rhs, out):
    return _Spec("u_mul_e", lhs, rhs, out)


def copy_e(edge, out):
    return _Spec("copy_e", edge, out)


def sum_(msg, out):
    return _Spec("sum", msg, out)


class Graph:
    def __init__(self, src, dst, n_nodes):
        self.src, self.dst, self.n_nodes = src, dst, n_nodes
        self.ndata, self.edata = {}, {}

    def number_of_edges(self):
        return len(self.src)

    def to(self, device):  # DGL returns a copy; the device is taken from the tensors
        g = Graph(self.src, self.dst, self.n_nodes)
        g.ndata, g.edata = dict(self.ndata), dict(self.edata)
        return g

    def apply_edges(self, spec):
        a, b, out = spec.names
        src, dst = self.src.to(self.ndata[a].device), self.dst.to(self.ndata[a].device)
        self.edata[out] = self.ndata[a][src] + self.ndata[b][dst]

    def update_all(self, msg, red):
        ref = next(iter(self.ndata.values()))
        src, dst = self.src.to(ref.device), self.dst.to(ref.device)
        if msg.kind == "u_mul_e":
            m = self.ndata[msg.names[0]][src] * self.edata[msg.names[1]]
        elif msg.kind == "copy_e":
            m = self.edata[msg.names[0]]
        else:
            raise NotImplementedError(msg.kind)
        out = torch.zeros(self.n_nodes, m.shape[1], dtype=m.dtype, device=m.device)
        out.index_add_(0, dst, m)  # sum of the messages arriving at each node
        self.ndata[red.names[1]] = out


def complete_graph(n_nodes=8, edge_dim=64):
    """The graph SoundAQnet uses: every ordered pair (i, j) is an edge, edge features are ones."""
    src = torch.arange(n_nodes).repeat_interleave(n_nodes)
    dst = torch.arange(n_nodes).repeat(n_nodes)
    g = Graph(src, dst, n_nodes)
    g.edata["feat"] = torch.ones(n_nodes * n_nodes, edge_dim)
    return g


def batch(graphs):
    src, dst, off = [], [], 0
    for g in graphs:
        src.append(g.src + off)
        dst.append(g.dst + off)
        off += g.n_nodes
    out = Graph(torch.cat(src), torch.cat(dst), off)
    for key in graphs[0].ndata:
        out.ndata[key] = torch.cat([g.ndata[key] for g in graphs])
    for key in graphs[0].edata:
        out.edata[key] = torch.cat([g.edata[key] for g in graphs])
    return out


def install():
    dgl = types.ModuleType("dgl")
    fn = types.ModuleType("dgl.function")
    fn.u_add_v, fn.u_mul_e, fn.copy_e, fn.sum = u_add_v, u_mul_e, copy_e, sum_
    dgl.function, dgl.batch = fn, batch
    sys.modules["dgl"], sys.modules["dgl.function"] = dgl, fn
