"""Small planning helpers for Chapter 43."""


# tag::tree-search[]
def tree_search(start, expand, value, depth, beam=2):
    """Keep the best partial plans under a learned or scripted value estimate."""
    frontier = [(start, [start])]
    best = (value(start), [start])
    for _ in range(depth):
        candidates = []
        for state, path in frontier:
            for child in expand(state):
                child_path = path + [child]
                candidates.append((value(child), child, child_path))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        frontier = [(state, path) for _, state, path in candidates[:beam]]
        if candidates[0][0] > best[0]:
            best = (candidates[0][0], candidates[0][2])
    return best[1]
# end::tree-search[]


# tag::plan-execute[]
def plan_then_execute(goal, planner, executor):
    observations = []
    for step in planner(goal):
        observations.append(executor(step))
    return observations
# end::plan-execute[]
