"""Constant-size semantic values for scalar expressions over binary images.

Each fact enumerates at most two independent binary frontiers (four points).
Arithmetic intermediates may be nonbinary. Movement and wider expressions
open new opaque frontiers; no image pixels or sample inputs are evaluated.
"""

from itertools import product

from .graph import Image


SCALAR = {"weighted", "combine", "saturating_weighted", "saturating_combine", "threshold"}


def reduce_support(frontier, values):
    """Remove variables whose two cofactors agree, retaining product order."""
    for position in reversed(range(len(frontier))):
        bit = 1 << (len(frontier) - position - 1)
        if all(values[index] == values[index | bit]
               for index in range(len(values)) if not index & bit):
            values = tuple(value for index, value in enumerate(values) if not index & bit)
            frontier = frontier[:position] + frontier[position+1:]
    return frontier, values


class BooleanValues:
    """Canonicalize proven Boolean values without increasing the search bound."""

    def __init__(self):
        self.facts, self.order, self.interned = {}, {}, {}
        self.anchor = None
        self.proofs = []
        self.counts = dict.fromkeys(("aliases", "constants", "cse"), 0)

    def fact(self, node):
        if node.lo == node.hi:
            return (), (node.lo,)
        if node.op in SCALAR:
            # A zero coefficient does not depend on its argument, even when
            # that argument has an unsupported wider domain.
            needed = (node.args if node.op == "threshold" else
                      tuple(arg for arg, weight in zip(node.args, node.params[:-1]) if weight))
            facts = [self.facts.get(arg) for arg in needed]
            if all(value is not None for value in facts):
                frontier = tuple(sorted({leaf for fact in facts for leaf in fact[0]},
                                        key=self.order.__getitem__))
                if len(frontier) <= 2:
                    values = []
                    for bits in product((0, 1), repeat=len(frontier)):
                        assignment = dict(zip(frontier, bits))
                        arguments = {}
                        for arg, (leaves, table) in zip(needed, facts):
                            index = 0
                            for leaf in leaves:
                                index = 2*index + assignment[leaf]
                            arguments[arg] = table[index]
                        if node.op == "threshold":
                            value = int(arguments[node.args[0]] >= node.params[0])
                        else:
                            value = node.params[-1] + sum(
                                weight*arguments[arg] for arg, weight in
                                zip(node.args, node.params[:-1]) if weight)
                            if node.op.startswith("saturating_"):
                                value = max(0, min(255, value))
                        if not node.lo <= value <= node.hi:
                            return None
                        values.append(value)
                    return reduce_support(frontier, tuple(values))
        # A movement result is its own variable. Composing through it would
        # erase observable crop/zero-padding boundaries.
        if (node.lo, node.hi) == (0, 1):
            return (node,), (0, 1)
        return None

    def rewrite(self, node, original_id):
        self.order.setdefault(node, len(self.order))
        if node.op == "input" and self.anchor is None:
            self.anchor = node
        fact = self.fact(node)
        self.facts[node] = fact
        if fact is None or not 0 <= node.lo <= node.hi <= 1:
            return node
        frontier, values = fact
        kind, result = None, node
        if len(frontier) == 1 and values == (0, 1) and frontier[0] is not node:
            kind, result = "aliases", frontier[0]
        elif fact in self.interned:
            kind, result = "cse", self.interned[fact]
        elif not frontier and node.op != "input":
            # One legal P generates a uniform Boolean plane. Keep an existing
            # canonical constant rather than rebuilding it or its dependencies.
            if node.op != "weighted" or node.params != (0, values[0]) or node.args[0] is not self.anchor:
                result = Image("weighted", (self.anchor,), (0, values[0]), values[0], values[0])
                kind = "constants"
                self.order[result] = len(self.order)
                self.facts[result] = fact
        self.interned.setdefault(fact, result)
        if kind is not None:
            self.counts[kind] += 1
            self.proofs.append({"original_node": original_id, "operation": node.op,
                                "parameters": list(node.params), "kind": kind,
                                "value_id_scope": "semantic pass, before instruction selection",
                                "frontier": [{"value_id": self.order[leaf], "operation": leaf.op,
                                              "parameters": list(leaf.params), "range": [0, 1]}
                                             for leaf in frontier],
                                "truth_table": list(values), "checked_points": len(values),
                                "proof_basis": "proven singleton range" if node.lo == node.hi else
                                               "complete binary truth signature, irrelevant variables removed",
                                "replacement_value": self.order[result]})
        return result

    def stats(self):
        return {"semantic_ir_rewrites": len(self.proofs),
                "semantic_ir_proofs": self.proofs,
                **{"semantic_ir_" + name: count for name, count in self.counts.items()}}
