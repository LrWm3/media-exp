"""Closed, small scalar graph cuts for bounded exact instruction synthesis.

Movement is a boundary, and named or externally used intermediates cannot be
removed. Proofs enumerate the full Cartesian input domain, not sample images.
"""

from itertools import product

from .graph import CompileError, Image
from .synthesize import synthesize


SCALAR = {"weighted", "combine", "saturating_weighted", "saturating_combine", "threshold"}
MAX_CUTS = 24
MAX_NODES = 32
MAX_POINTS = 64


class ExactSearch:
    def __init__(self, graph, named, baseline):
        self.graph, self.named, self.baseline = graph, named, baseline
        self.ids = {node: index for index, node in enumerate(graph)}
        self.users = {node: set() for node in graph}
        for node in graph:
            for arg in node.args:
                self.users[arg].add(node)
        self.cuts = {}
        for node in graph:
            own = (self.ids[node],)
            candidates = {own}
            if node.op in SCALAR:
                choices = [((self.ids[arg],),) if arg in named else self.cuts[arg] for arg in node.args]
                for parts in product(*choices):
                    leaves = tuple(sorted(set().union(*parts)))
                    if 1 <= len(leaves) <= 2:
                        candidates.add(leaves)
            # Prefer deeper cuts, but always retain the node itself. The bounds
            # cap compile work even for large or heavily reconvergent graphs.
            rest = sorted(candidates - {own}, key=lambda cut: (min(cut), len(cut), cut))
            self.cuts[node] = (own, *rest[:MAX_CUTS - 1])
        self.attempts = 0

    def interior(self, root, frontier):
        leaves, inside, pending = set(frontier), set(), [root]
        while pending:
            node = pending.pop()
            if node in leaves or node in inside:
                continue
            if node.op not in SCALAR or (node is not root and node in self.named):
                return None
            inside.add(node)
            if len(inside) > MAX_NODES:
                return None
            pending.extend(node.args)
        if any(self.users[node] - inside for node in inside if node is not root):
            return None
        return sorted(inside, key=self.ids.__getitem__)

    @staticmethod
    def truth_table(frontier, inside):
        target = []
        for values in product(*(range(node.lo, node.hi + 1) for node in frontier)):
            evaluated = dict(zip(frontier, values))
            for node in inside:
                args = [evaluated[arg] for arg in node.args]
                if node.op == "threshold":
                    value = int(args[0] >= node.params[0])
                else:
                    value = sum(a * w for a, w in zip(args, node.params[:-1])) + node.params[-1]
                    if node.op.startswith("saturating_"):
                        value = max(0, min(255, value))
                if not node.lo <= value <= node.hi:
                    return None
                evaluated[node] = value
            target.append(evaluated[inside[-1]])
        return tuple(target)

    def choose(self, root):
        if root.op not in SCALAR or (root.lo, root.hi) != (0, 1):
            return None
        best = None
        for cut in self.cuts[root][1:]:
            frontier = tuple(self.graph[index] for index in cut)
            points = 1
            for leaf in frontier:
                points *= leaf.hi - leaf.lo + 1
            if points > MAX_POINTS or any(not 0 <= leaf.lo <= leaf.hi <= 255 for leaf in frontier):
                continue
            inside = self.interior(root, frontier)
            if not inside:
                continue
            # Fresh input nodes stop existing fusion at exactly this frontier.
            sources = tuple(Image("input", (), (f"cut_{i}",), leaf.lo, leaf.hi)
                            for i, leaf in enumerate(frontier))
            copies = dict(zip(frontier, sources))
            for node in inside:
                copies[node] = Image(node.op, tuple(copies[arg] for arg in node.args),
                                     node.params, node.lo, node.hi)
            try:
                cost = len(self.baseline(sources, {"output": copies[root]})[0])
            except CompileError:
                continue
            if cost < 2:
                continue
            target = self.truth_table(frontier, inside)
            if target is None:
                continue
            self.attempts += 1
            sequence = synthesize(tuple((leaf.lo, leaf.hi) for leaf in frontier), target,
                                  max_instructions=min(3, cost - 1))
            if sequence is None:
                continue
            score = (cost - len(sequence), -len(sequence), -len(frontier))
            if best is None or score > best[0]:
                proof = {"graph_id": self.ids[root],
                         "frontier": [{"graph_id": self.ids[leaf], "range": [leaf.lo, leaf.hi]}
                                      for leaf in frontier],
                         "checked_points": points, "truth_table": list(target),
                         "original_instructions": cost,
                         "replacement_instructions": [
                             {"inputs": list(step.inputs), "weights": list(step.weights),
                              "bias": step.bias, "denominator": step.denominator}
                             for step in sequence]}
                best = score, frontier, sequence, proof
        return None if best is None else best[1:]
