"""Independent checker for binary, unit-propagating UNSAT trees.

No solver code is imported. A successful check certifies UNSAT only for the
supplied clauses together with the supplied assumptions. In particular, a
locally assumed literal must not be silently promoted to a global fact.

``work`` is a deterministic logical-operation counter, not CPU time:
* one header and one operation per literal for initial clause validation;
* one inspection per assumption and one assignment per newly assigned variable;
* one node dispatch and one operation per unit-record validation;
* one header and one operation per literal on each cited-clause inspection;
* one operation per unit assignment and per branch validation;
* for each child, one operation per copied assignment plus one branch assignment.
It includes validation of unused input clauses. Python container allocation,
hashing, dependency-set maintenance, and final sorting are not modeled.
Invalid checks raise ProofError, whose ``work`` includes work charged up to
rejection. Only a valid check returns a dependency set.
"""

from __future__ import annotations

from typing import Any


class ProofError(ValueError):
    """Malformed or invalid proof, with the verification work already spent."""

    def __init__(self, message: str, work: int):
        super().__init__(message)
        self.work = work


def check_unsat(
    clauses: dict[str, list[int]], assumptions: list[int], proof: dict
) -> dict:
    """Validate a complete UNSAT tree and return its actual clause dependencies.

    Each node has ``units`` and exactly one terminal form: ``conflict``, or
    ``branch``, ``left``, ``right``. Branch variables are positive integers,
    unassigned at that node. The left child assigns False, the right True.
    Clause literals and assumptions are nonzero integers (booleans rejected).
    Repeated clause literals are permitted as input, but a cited unit reason
    must contain exactly one unassigned literal occurrence. A duplicated unit
    literal therefore requires normalization before serving as a unit reason.
    Repeated consistent assumptions are permitted; contradictory ones are not.
    """
    work = 0
    dependencies: set[str] = set()
    active_nodes: set[int] = set()

    def reject(message: str) -> None:
        raise ProofError(message, work)

    def integer(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    if not isinstance(clauses, dict):
        reject("clauses must be a dictionary")
    for clause_id, clause in clauses.items():
        work += 1
        if not isinstance(clause_id, str) or not isinstance(clause, list):
            reject("clause IDs must be strings and clauses must be lists")
        for literal in clause:
            work += 1
            if not integer(literal) or literal == 0:
                reject("clause literals must be nonzero integers")
    if not isinstance(assumptions, list):
        reject("assumptions must be a list")
    assignment: dict[int, bool] = {}
    for literal in assumptions:
        work += 1
        if not integer(literal) or literal == 0:
            reject("assumptions must be nonzero integers")
        variable, value = abs(literal), literal > 0
        if variable in assignment:
            if assignment[variable] != value:
                reject("contradictory assumptions are not an accepted proof")
        else:
            work += 1
            assignment[variable] = value

    def inspect(clause_id: Any, values: dict[int, bool]) -> tuple[bool, list[int]]:
        nonlocal work
        work += 1
        if not isinstance(clause_id, str) or clause_id not in clauses:
            reject("unknown or malformed clause reference")
        dependencies.add(clause_id)
        satisfied = False
        unassigned: list[int] = []
        for literal in clauses[clause_id]:
            work += 1
            variable = abs(literal)
            if variable not in values:
                unassigned.append(literal)
            elif values[variable] == (literal > 0):
                satisfied = True
        return satisfied, unassigned

    def visit(node: Any, values: dict[int, bool]) -> None:
        nonlocal work
        work += 1
        if not isinstance(node, dict):
            reject("each proof node must be a dictionary")
        node_id = id(node)
        if node_id in active_nodes:
            reject("cyclic proof object")
        keys = set(node)
        conflict_form = keys == {"units", "conflict"}
        branch_form = keys == {"units", "branch", "left", "right"}
        if not conflict_form and not branch_form:
            reject("node must have exactly one complete terminal form")
        if not isinstance(node["units"], list):
            reject("units must be a list")
        active_nodes.add(node_id)
        try:
            for entry in node["units"]:
                work += 1
                if not isinstance(entry, list) or len(entry) != 2:
                    reject("unit record must be [literal, reason_clause_id]")
                literal, clause_id = entry
                if not integer(literal) or literal == 0:
                    reject("asserted unit must be a nonzero integer")
                variable = abs(literal)
                if variable in values:
                    reject("unit variable is already assigned")
                satisfied, unassigned = inspect(clause_id, values)
                if satisfied or unassigned != [literal]:
                    reject("cited reason does not force exactly the asserted unit")
                work += 1
                values[variable] = literal > 0
            if conflict_form:
                satisfied, unassigned = inspect(node["conflict"], values)
                if satisfied or unassigned:
                    reject("cited conflict is not completely false")
                return
            work += 1
            variable = node["branch"]
            if not integer(variable) or variable <= 0 or variable in values:
                reject("branch variable must be positive and unassigned")
            for key, value in (("left", False), ("right", True)):
                work += len(values) + 1
                child_values = values.copy()
                child_values[variable] = value
                visit(node[key], child_values)
        finally:
            active_nodes.remove(node_id)

    visit(proof, assignment)
    return {"ok": True, "dependencies": sorted(dependencies), "work": work}


def checker_selftest() -> dict:
    """Small independent valid/tampered checks, including assumption scope."""
    from copy import deepcopy

    accepted = 0
    rejected = 0
    total_work = 0

    def accept(clauses: dict, assumptions: list, proof: dict, expected: list) -> dict:
        nonlocal accepted, total_work
        result = check_unsat(clauses, assumptions, proof)
        assert result["dependencies"] == expected, result
        assert result["ok"] is True and result["work"] > 0
        accepted += 1
        total_work += result["work"]
        return result

    def refuse(clauses: Any, assumptions: Any, proof: Any) -> None:
        nonlocal rejected, total_work
        try:
            check_unsat(clauses, assumptions, proof)
        except ProofError as error:
            rejected += 1
            total_work += error.work
        else:
            raise AssertionError("tampered proof was accepted")

    empty = {"units": [], "conflict": "empty"}
    result = accept({"empty": []}, [], empty, ["empty"])
    assert result["work"] == 3  # input header, node, conflict header
    units = {"units": [[1, "a"]], "conflict": "not_a"}
    result = accept({"a": [1], "not_a": [-1]}, [], units, ["a", "not_a"])
    assert result["work"] == 11

    clauses = {"ab": [1, 2], "anb": [1, -2], "nab": [-1, 2], "nanb": [-1, -2]}
    tree = {
        "units": [], "branch": 1,
        "left": {"units": [[2, "ab"]], "conflict": "anb"},
        "right": {"units": [[2, "nab"]], "conflict": "nanb"},
    }
    accept(clauses, [], tree, ["ab", "anb", "nab", "nanb"])
    tampered = deepcopy(tree)
    tampered["right"]["conflict"] = "nab"
    refuse(clauses, [], tampered)
    tampered = deepcopy(tree)
    tampered["left"]["units"][0][0] = -2
    refuse(clauses, [], tampered)
    tampered = deepcopy(tree)
    tampered["left"]["units"].append([2, "ab"])
    refuse(clauses, [], tampered)
    refuse(clauses, [1], tree)
    refuse(clauses, [1, -1], tree)
    refuse(clauses, [], {"units": [], "branch": 1, "left": empty})
    refuse(clauses, [], {"units": [], "conflict": "missing"})
    refuse({"x": [True]}, [], empty)
    refuse({"x": [0]}, [], empty)
    refuse({"x": [1]}, [True], empty)
    refuse({"x": [1]}, [], {"units": [[True, "x"]], "conflict": "x"})
    refuse({"x": [1]}, [], {"units": [], "conflict": "x", "extra": 1})

    # F=(a or b), under assumptions not-a and not-b is UNSAT. Dropping
    # not-a destroys that certificate; local b is not a global fact.
    scoped = {"units": [], "conflict": "ab"}
    accept({"ab": [1, 2], "unused": [3]}, [-1, -2], scoped, ["ab"])
    refuse({"ab": [1, 2]}, [-2], scoped)
    accept({"ab": [1, 2]}, [-1, -1, -2], scoped, ["ab"])

    cyclic = {"units": [], "branch": 1, "left": None, "right": empty}
    cyclic["left"] = cyclic
    refuse({"empty": []}, [], cyclic)
    return {"ok": True, "accepted": accepted, "rejected": rejected, "work": total_work}


if __name__ == "__main__":
    import json

    print(json.dumps(checker_selftest(), sort_keys=True))
