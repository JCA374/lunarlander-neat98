# Lunar Lander in wind, turbulence and a boulder: 98% by Wednesday (brief for Claude Code)

> **How to use this file:** copy it into a new, empty folder as `BRIEF.md`, open Claude Code there and say
> *"Read BRIEF.md and start."* Everything Claude needs is in this file plus the previous project folder it points to.

**From the user:** I tried NEAT for two days in `../lunarlander-curriculum` and it stalled at about 85%. This time
you choose the method. Same task, same goal, same deadline. I like curriculum learning, so use it where it helps. I want
a homepage with logs and lots of podcasts, so I can follow along without being at the laptop. You have full autonomy:
change the curriculum, the rewards, the model, the starting point or the language as you see fit. Be smart, test
things, and change course when the numbers say so. Report honestly, including failures.

---

## 1. The goal and when we stop

- **Done when:** the agent lands **≥ 98% of 1,000 unseen flights** on the real task (§2). Report it with a 95%
  Wilson interval, and check it again on a second, fresh block of 1,000 test seeds before you call it done. Then stop.
- **Or stop at:** **Wednesday 2026-11-04, 17:00** (local time), whatever the result. At **16:27** start the final
  wrap-up (§9), so it's finished by 17:00.
- No result counts unless it was measured on fresh seeds: ≥ 100 flights for any claim, 1,000 for the goal.

## 2. The task (identical to the previous project, so results compare)

**Use the previous project's environment code unchanged:** copy `../lunarlander-curriculum/lander/env.py` (and
`lander/route_params.json`, which it reads) and treat it as frozen. Add a test that records its sha256, so nobody
changes the task by accident. It defines:

- Gymnasium **LunarLander-v2** (`gymnasium[box2d]==0.29.1`, `numpy<2`), **discrete** actions (4), 1,000 steps max.
- **Wind** `wind_power = 15` and **turbulence** `turbulence_power = 1.5`, seeded per flight with `seed_wind`.
  Gymnasium takes the gust phase from the global RNG; without `seed_wind`, one seed doesn't mean one flight.
- **The boulder**: a static circle (radius 0.8–1.3 m) in the air between the start and the pad, placed per seed by
  `boulder_spec`. Its top is always ≥ 3.5 m below the start; the user approved this fairer layout on 2026-10-30.
  Touching the boulder (body or leg) is a crash.
- **"Landed"** = came to rest (the env's +100 terminal) **between the flags**. A crash, a timeout, a boulder hit or an
  off-pad landing is a failure. The real task (`GOAL_WORLD`) has no help: no training wheels, no steady push.
- **What the agent may see:** the 8 standard observations plus sensors computed **only from past observations and its
  own past actions** (the existing 16 inputs in `inputs()` are fine: drift, |x|, boulder dx/dy/r, waypoint dx,
  above-boulder, clean wind estimate). **No cheating:** the policy must never read hidden env state (`wind_idx`,
  `torque_idx`, the seed, future wind). Write a test that the policy's inputs come only from `inputs()`.
- **Seed ranges** (keep them separate, tested): train 0+, cloning/demos 5e8+, stage checks 1e9+, selection 1.5e9+,
  **goal test 2e9+** (second block 2e9 + 10,000), analysis 3e9+.

**Where the old project stood** (all on the real task, fair boulder layout, 100-flight checks unless noted):

| Agent | Landed |
|---|---|
| NEAT run K champion (gentle mutation, seeded from a cloned pilot), gens 0–187 | ~84–85% average, best single check 91% |
| Hand-written pilot v2 (CMA-ES-tuned gains, clean wind sensor) | 86.3% on the *old* layout; re-measure on the fair one first |
| Hand-written pilot v1 | 71.7% |
| Crude "fly around the boulder" escape: how often the layout itself looks unwinnable | 0.65% of seeds |

So the ceiling is roughly 99%+, and 98% needs a real jump over everything we have.

## 3. Choosing the method: you decide, but measure first

You may pick any model and algorithm. Before committing, **spend at most the first ~4 hours on a fair bake-off**: give
2–3 candidates the same CPU budget (say 45 minutes each) and compare them on the same 200 check seeds. Write the choice
and the numbers in the journal and in a podcast episode.

Candidates worth considering, from what we learned (you're free to pick others):

- **PPO** (or a similar on-policy RL method) with many parallel envs, a 2×64 or 2×128 tanh MLP, the 16 inputs, and
  observation normalisation. Gradient methods use every step of experience; NEAT used only one number per flight.
- **CMA-ES or OpenAI-ES** over an MLP, warm-started from the cloned pilot (`../lunarlander-curriculum/models/seed_mlp16.json`,
  which lands ~85%). Evolution strategies move all weights a little at a time, which is exactly what NEAT couldn't do.
- **Better teacher, then distil:** improve the hand-written pilot (or a planner that only uses the agent's own
  observations) and behaviour-clone it with DAgger (`lander/clone.py` already does this), then fine-tune with RL or ES.
- **A small memory** (GRU or a few stacked frames) if failure analysis shows the agent can't tell gusts apart from its
  own thrust.

Whatever you choose: **start from the best existing agent when you can** (warm start), and check right away that the
warm-started agent really is as good as its source on the real task (in the old project a conversion bug once made a
"clone" pick the wrong action one step in nine).

## 4. Curriculum: the user wants it, and here is what we learned about it

The user likes curriculum learning, so design one (stages, rewards, pass checks) and show it on the site **before** each
stage starts: world, reward, pass check, generation/step cap and expected outcome. You set up and change the steps
and rewards as you go.

What the old project measured (use it):

- **Easy stages can erase a good starting agent.** A good start trained on calm or easy stages lost skill on the real
  task: 90% → 52% in one run, 83% → 58% in another. The clone relies on the wind sensor, and in calm air the sensor
  reads ~0, so those weights drift freely. If you start from a strong agent, make the curriculum **anti-forgetting**:
  always mix in real-task flights (rehearsal, 30–50%), and pass checks always fly the real task too.
- **Curriculum works best from a weak start**: calm → steady push → gusts ramped 5→10→15 → turbulence ramped →
  boulder (shrunk, then full size). Fading **training wheels** (a stabilising torque that fades to 0 within a stage)
  worked better than heavy reward shaping. Checks always run without them.
- **A hard-case curriculum** (oversample the seeds the agent fails: strong gusts, boulder right over the pad) is the
  natural kind for the last few percent. The failure analysis tools are in `experiments/failures.py` and `boulder_map.py`.
- **Reward traps we found:** the game score rewards fuel saving and centring, so "points up, landings down" happened
  (score 182 → 220 while landings fell). The fitness that worked pays mainly for landing: 0.3 × game score + 300
  per landing − 200 per timeout − 100 per boulder hit, no centre bonus. Also watch for hovering until the timeout, deliberate
  off-pad landings and fast touchdowns. Check the timeout rate, the off-pad rate and the touchdown speed at every check-in.

## 5. Use this laptop well, and make the code fast once it works

**First, measure the machine** (`nproc`, `free -h`, `lscpu`, a GPU check) and write it on the site. Last time: Intel
i5-8250U, 4 cores / 8 threads, 7.6 GB RAM, **no usable GPU** (Intel UHD 620). Plan for CPU: small networks, many
parallel envs, `torch.set_num_threads` tuned, and no oversubscription (workers × threads ≤ 8). Narration and video run with `nice`.

**Then profile before optimising.** What the old project measured per simulated step (117 µs): Box2D itself only
1.5 µs, the gymnasium Python wrapper and our Python wrapper most of the rest, and the network evaluation 70% of the
time until it was compiled to NumPy (111 s → 47 s per generation). Bottlenecks are in Python glue, not physics.

**Once the method works, make the hot path fast** (Rust, C++ or Go, whatever fits; Rust via PyO3/maturin or C++ via
pybind11 are natural choices):

- Likely wins: the policy forward pass batched over all envs, the sensors (`inputs()`, `WindSensor`), and above all a
  **native vectorised re-implementation of the lander physics** (Box2D in C++/Rust, or a careful port) stepping hundreds
  of envs per call.
- **The rule that keeps it honest:** the Python gymnasium env in §2 is the reference. A fast env may be used for
  **training only**, after a test shows it agrees with the reference: same seeds, same actions → same outcomes on
  ≥ 98% of 1,000 seeds, and the same agent's landed rate within the confidence interval. **All checks and the final goal
  test run in the reference env.**
- Speedups count only if measured end-to-end (flights per second, training time to a fixed score), before and after,
  written in the journal.

## 6. Reuse from the previous project (`../lunarlander-curriculum`)

Copy and adapt rather than rewrite. Same logic, publisher, checker, tester and evaluator:

| What | File(s) |
|---|---|
| Task, sensors, flight runner | `lander/env.py`, `lander/route_params.json` (frozen) |
| Hand-written pilot + tuned gains | `lander/teacher.py`, `lander/teacher_params.json` (v2), `experiments/tune_teacher.py` (CMA-ES tuning) |
| Behaviour cloning / DAgger, the cloned pilot | `lander/clone.py`, `models/seed_mlp16.json` |
| Goal check (1,000 test flights, Wilson CI, failed seeds) | `experiments/goal_check.py` (adapt the policy loader) |
| Paired comparison with bootstrap CI | `experiments/paired.py` |
| Failure analysis | `experiments/failures.py`, `experiments/boulder_map.py` |
| Journal (shown on the site) | `experiments/journal.py` → `experiments/journal.jsonl` |
| Curriculum as JSON stages (world, ramps, support, reward, pass, cap, rehearsal) | `curricula/K.json`, the stage logic in `train.py` |
| Training run pieces: checkpoint/resume, SIGTERM checkpoint, events.jsonl, stats.csv, champion archive, override.json | `train.py` (`Run` class) |
| Supervisor that restarts a crashed run; safe restart | `supervise.sh`, `restart_run.sh` |
| Site builder + 15-minute publish loop | `publish.py`, `publish_loop.sh`, `docs/` (index, app.js, style.css, lessons, network, senses, podcast) |
| Local live dashboard | `dashboard.py` (http://localhost:8050) |
| Tests | `tests/test_core.py` (determinism, boulder crash, seed ranges, resume, clone equivalence, ...) |
| Podcast settings | `podcast.conf` (change REPO, SITE) |
| Working method | `METHOD.md` (2-hour check-ins, one change at a time, verification, page, podcast) |

## 7. The homepage (GitHub Pages) and lots of podcasts

- **Repo and site:** create a new public repo (suggested `JCA374/lunarlander-98`; ask the user before creating it),
  Pages from `main:/docs`, link the site from the README and the repo homepage. Deployment counts only when
  `gh api repos/<repo>/pages/builds/latest` says `built <sha>` and the pages return HTTP 200.
- **Tabs:** *Log* (newest first: every journal note, stage start/pass, goal check, new champion, each with a replay GIF
  of the agent as it was then; a goal table and a countdown to Wednesday 17:00 at the top; a progress chart with an
  autoscaled y-axis and fixed-height chart boxes so they don't grow), *Curriculum* (each stage with its reward and expected outcome, set
  before it starts), *How it sees* (the inputs vs. reality), *The model* (very educational, with lots of examples: what
  the network is, how it learns, how it changed over time), *Speed* (the machine, profiles, before/after
  timings of each optimisation), *Lessons*, *Podcast*. The user prefers too much explanation over too little.
- **Publish every 15 minutes** automatically (the publish loop), versioned script URLs so browsers don't show stale pages.
- **Podcast: lots of episodes.** Use the user-level skill `project-podcast` (`podcast.conf` in the project). Make one for
  **every major event**: the plan (episode 1), the machine and the bake-off, the method chosen, each curriculum stage
  passed, each breakthrough, each failure or trap found, each speed-up, each correction of an earlier claim, and the
  final result. 2–6 minutes, first person, every number from the logs, written for the ear. No episodes for
  "still training".

## 8. How to run the work (as before; details in METHOD.md)

1. **Day 1, in order:** measure the machine → set up the repo, `.venv`, tests (determinism first: same seed twice → the
   same flight) → copy the frozen env → re-measure the baselines in §2 on 1,000 test seeds → site + episode 1 → the bake-off
   (§3) → start the chosen run under the supervisor.
2. **Cron jobs** (session-only, so set them up again after any restart): a check-in every 2 hours, and the final wrap-up
   one-shot at Wednesday 16:27. At each check-in: look (processes, stats, memory, load), judge (improving, plateau,
   trap), decide (one change at a time, reason written down *before* the result), record (journal), publish, and report
   a few lines in the chat. If nothing changed, say so in one line.
3. **Verification rules:** pass checks on fresh check seeds with a confirmation on a second fresh set; champions
   picked on few flights are flattered (the winner's curse, seen in every project so far), so re-fly the top candidates
   on more selection seeds; claim "better" only from a paired comparison on ≥ 100 shared seeds with a bootstrap CI.
4. **Archive every champion**, never overwrite a verified best. Keep a `models/best/` with its goal-check JSON.
5. **Correct mistakes in public**, in the journal and the next episode.
6. **Background jobs:** `setsid nohup`, SIGTERM must checkpoint, a supervisor restarts crashes (a run once died overnight
   and lost 11 hours). Never `pkill -f` a pattern that's also in your own command line: use anchored patterns
   (`pgrep -f "^python train.py"`) and kill by PID. Kill test subprocesses by process group, so no orphaned trainers
   eat the CPU (that once ran unnoticed for 7.5 hours).
7. **Ask the user** only for things that change the task or go public (a new repo, a fairer world). Everything else is
   yours to decide.

## 9. Final wrap-up (start Wednesday 16:27, or as soon as 98% is verified)

Stop the run cleanly (SIGTERM → checkpoint), the publish loop, the dashboard and the cron jobs. Run the goal check on the
best agent: 1,000 test seeds (+ the second block if there's time), in the reference env. Save the agent in `models/best/`
with its results. Write `CONTINUE.md` (the result with its CI, exact resume commands, what worked, what didn't, open
ideas), a final journal entry, a final podcast episode, a last site update, commit and push. Then tell the user in the chat.

## 10. Lessons from the NEAT project, in one list

- Sensing beats network size: the clean wind sensor (subtract each action's known push; it correlates 0.97 with the
  true wind, the old drift sensor 0.14) lifted the pilot from 75% to 86%. Look for missing information before adding capacity.
- Check that the task is fair: about 2% of boulder layouts were unwinnable until the layout was fixed.
- A warm start is only worth it if training doesn't destroy it: NEAT's structural and weight-replacement mutations
  wrecked trained networks (children averaged 70.5% from an 86% parent; gentle settings 83.4%). Measure what one
  update step does to a good agent.
- Few flights per evaluation make selection lucky: wind noise is large; 24+ flights per candidate plus re-flying the top 8.
- Determinism and resume bugs cost the most time: test them first (seeded wind, the RNG, node/ID counters after resume).
- Easy curriculum stages can make a strong agent forget (rehearse the real task).
- Pay for what you want (landings), not for a proxy (game score).
- Profile before rewriting: the physics was never the bottleneck.
