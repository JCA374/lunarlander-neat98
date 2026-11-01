# Lander NEAT 98

**🌐 Live site: https://jca374.github.io/lunarlander-neat98/**

NEAT (evolved networks, weights and wiring) learns to land Gymnasium's Lunar Lander in wind 15, turbulence 1.5 and past a
floating boulder (layout v3, passable by construction). Goal: ≥ 98% landed between the flags on 1,000 unseen flights,
confirmed on a second 1,000, by Friday 2026-11-06 17:00. Clean start, curriculum, fast C++ simulator for training only;
all checks run in the reference Python environment.

- `BRIEF.md`: the task and the rules. `lander/`: the frozen task (sha256 in `tests/test_core.py`).
- `docs/`: the site, rebuilt every 15 minutes by `publish_loop.sh`.
- Tests: `.venv/bin/python -m pytest -q`
