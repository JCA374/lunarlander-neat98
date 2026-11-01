"""Seed ranges. Never mix them up (tests/test_core.py checks they don't overlap)."""
TRAIN = 0                    # training flights (RL rollouts, ES evaluations)
CLONE = 500_000_000          # behaviour cloning / DAgger demos
CHECK = 1_000_000_000        # stage pass checks, each block used once
SELECT = 1_500_000_000       # choosing champions / re-flying candidates
TEST = 2_000_000_000         # goal test: block 1 = TEST + 0.., block 2 = TEST + 10_000..; baselines TEST + 50_000..
ANALYSIS = 3_000_000_000     # comparisons, failure analysis, bake-off checks
SEED_RANGES = {'train': (TRAIN, CLONE), 'clone': (CLONE, CHECK), 'check': (CHECK, SELECT),
               'select': (SELECT, TEST), 'test': (TEST, ANALYSIS), 'analysis': (ANALYSIS, 4_000_000_000)}
GOAL_BLOCK_1, GOAL_BLOCK_2, BASELINE_BLOCK = TEST, TEST + 10_000, TEST + 50_000
