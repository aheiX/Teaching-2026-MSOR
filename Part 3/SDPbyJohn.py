from __future__ import annotations
from dataclasses import dataclass, replace
from collections import deque
from typing import Dict, List, Tuple, FrozenSet, Optional

# ------------------ World definition ------------------

ROOM1 = "room1"
ROOM2 = "room2"
ROOM3 = "room3"
ROOM4 = "room4"
ROOM5 = "room5"
OUTSIDE1 = "outside1"
OUTSIDE2 = "outside2"

ITEM1 = "item1"
ITEM2 = "item2"

# FIX: allow room2 <-> room3 movement (so you can go from room2 to room3 directly)
GRAPH: Dict[str, List[str]] = {
    ROOM1: [ROOM2, ROOM3],
    ROOM2: [ROOM1, ROOM3],                  # <-- added ROOM3
    ROOM3: [ROOM1, ROOM2, OUTSIDE1, OUTSIDE2, ROOM4],  # <-- added ROOM2
    ROOM4: [ROOM3, OUTSIDE2, ROOM5],
    ROOM5: [ROOM4],
    OUTSIDE1: [ROOM3],
    OUTSIDE2: [ROOM3, ROOM4],
}

ITEM_START_POS = {
    ITEM1: ROOM2,  # first item is in room2
    ITEM2: ROOM5,  # second item is in room5
}

# ------------------ State ------------------

@dataclass(frozen=True)
class State:
    dog_pos: str
    remaining_items: FrozenSet[str]  # items not found yet

def items_in_room(room: str, remaining_items: FrozenSet[str]) -> List[str]:
    """Which (still missing) items are located in this room?"""
    return [it for it in remaining_items if ITEM_START_POS[it] == room]

def auto_pickup(s: State) -> Tuple[State, List[str]]:
    """
    Automatically pick up any remaining items in the current room.
    Returns (new_state, found_items).
    """
    found = items_in_room(s.dog_pos, s.remaining_items)
    if not found:
        return s, []
    new_remaining = frozenset(x for x in s.remaining_items if x not in found)
    return replace(s, remaining_items=new_remaining), found

def initial_state(start_pos: str = ROOM1) -> State:
    s = State(dog_pos=start_pos, remaining_items=frozenset(ITEM_START_POS.keys()))
    s, _ = auto_pickup(s)  # auto-pickup if you start on an item
    return s

# ------------------ Actions (move only) ------------------

def possible_actions(s: State) -> List[Tuple[str, State]]:
    actions: List[Tuple[str, State]] = []
    for nxt in GRAPH[s.dog_pos]:
        moved = replace(s, dog_pos=nxt)
        moved, _ = auto_pickup(moved)  # auto-pickup on entering
        actions.append((f"move {s.dog_pos} -> {nxt}", moved))
    return actions

# ------------------ Goal ------------------

def is_goal(s: State) -> bool:
    return len(s.remaining_items) == 0

# ------------------ BFS planner (shortest plan) ------------------

def bfs_plan(start: State) -> Optional[List[str]]:
    q = deque([start])
    prev: Dict[State, Optional[State]] = {start: None}
    prev_action: Dict[State, Optional[str]] = {start: None}

    while q:
        cur = q.popleft()
        if is_goal(cur):
            plan: List[str] = []
            s = cur
            while prev[s] is not None:
                plan.append(prev_action[s])  # type: ignore[arg-type]
                s = prev[s]  # type: ignore[assignment]
            plan.reverse()
            return plan

        for action, nxt in possible_actions(cur):
            if nxt not in prev:
                prev[nxt] = cur
                prev_action[nxt] = action
                q.append(nxt)

    return None

# ------------------ Text UI ------------------

def pretty_state(s: State) -> str:
    remaining = ", ".join(sorted(s.remaining_items)) if s.remaining_items else "none (all found)"
    return f"Dog: {s.dog_pos} | Remaining: {remaining}"

def play_interactive(start: State) -> None:
    s = start
    print("=== Dog finds 2 items (auto-pickup) ===")
    print("Goal: find both items.")
    print(pretty_state(s))
    print()

    while True:
        if is_goal(s):
            print("Nice Work! You win! Both items were found. Who's a Good Boy? 🐶")
            break

        acts = possible_actions(s)

        print("Actions:")
        for i, (a, _) in enumerate(acts, 1):
            print(f"  {i}. {a}")
        print("  a. auto-solve (shortest plan from here)")
        print("  q. quit")

        choice = input("> ").strip().lower()
        print()

        if choice == "q":
            print("Quit.")
            break

        if choice == "a":
            plan = bfs_plan(s)
            if plan is None:
                print("❌ No plan found.")
                continue
            print("Auto plan:")
            for step in plan:
                print(" -", step)

            # Execute plan with found-item messages
            for step in plan:
                # find the matching transition
                next_states = [(a, ns) for a, ns in possible_actions(s) if a == step]
                if not next_states:
                    raise RuntimeError(f"Plan step not applicable: {step}")
                _, ns = next_states[0]

                # Print found items by comparing remaining sets
                found_now = sorted(s.remaining_items - ns.remaining_items)
                s = ns
                if found_now:
                    for it in found_now:
                        print(f"Yayy! You found {it} in {s.dog_pos}!")

            print("\nEnd state:")
            print(pretty_state(s))
            print()
            continue

        if not choice.isdigit():
            print("Please enter a number, 'a', or 'q'.\n")
            continue

        idx = int(choice) - 1
        if idx < 0 or idx >= len(acts):
            print("Invalid choice.\n")
            continue

        # Execute chosen move, with found-item messages
        old = s
        s = acts[idx][1]
        found_now = sorted(old.remaining_items - s.remaining_items)
        if found_now:
            for it in found_now:
                print(f"🎉 You found {it} in {s.dog_pos}!")
        print(pretty_state(s))
        print()

def main():
    start = initial_state(start_pos=ROOM1)
    play_interactive(start)

if __name__ == "__main__":
    main()
