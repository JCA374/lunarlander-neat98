# NEAT lander, done right: 98% in wind, turbulence and past a fair boulder (brief for Claude Code)

> **How to use this file:** copy it into a new, empty folder as `BRIEF.md`, open Claude Code there and say
> *"Read BRIEF.md and start."* Everything Claude needs is in this file plus the folder of the previous project,
> `../lunarlander-98`, which it points to. (The project before that, `../lunarlander-curriculum`, has the old NEAT code.)

**From the user:** The last project reached the goal with PPO (98.8% and 98.4% on two blocks of 1,000 unseen flights).
Now I want the same result with **NEAT**, using everything we learned: a **clean start** (no copied pilot), a **smart
curriculum**, a **network that is complex enough**, and **fast code** for the hot path. Same task, same goal, same
standards of proof. One change to the world: make it explicit and tested that **every layout is passable**: the
boulder never sits so close to the start or the pad that it blocks the way. Homepage with logs, a film and lots of
podcasts, so I can follow along away from the laptop. Report honestly, including failures.

---

## 1. Goal and when to stop

- **Done when:** the NEAT champion lands **≥ 98% of 1,000 unseen flights** on the real task (§2), with a 95% Wilson
  interval, **and again on a second fresh block of 1,000**. Then stop and wrap up (§10).
- **Deadline:** ask the user for the date and time at the start (the last two projects had about 5 days). Set a cron
  job for the wrap-up 33 minutes before it.
- A result counts only if it was measured on fresh seeds: ≥ 100 flights for any claim, ≥ 1,000 for choosing the
  final champion (selection range), and the two goal blocks (test range) flown **once**, by the chosen champion only.

## 2. The task: the same as before, with a fair boulder layout (v3)

Copy `../lunarlander-98/lander/env.py`, `route_params.json` and `teacher.py` (env.py's route sensors call
`teacher.target_x`) and freeze them, **with one approved change: `boulder_spec` v3** (below). Record the sha256 of all
three in a test. Everything else is unchanged:

- Gymnasium **LunarLander-v2** (`gymnasium[box2d]==0.29.1`, `numpy<2`), 4 discrete actions, 1,000 steps max.
- Wind `wind_power = 15`, turbulence `turbulence_power = 1.5`, seeded per flight with `seed_wind` (gymnasium takes
  the gust phase from the global RNG; without `seed_wind` one seed is not one flight).
- **Landed** = came to rest (the env's +100) **between the flags** (|x| ≤ 0.2 in observation units = ±2 m around the
  pad centre). A crash, a boulder touch (body or leg), a timeout or a rest outside the flags is a failure. The real
  task (`GOAL_WORLD`) has no help.
- **What the pilot may see:** the 16 inputs of `inputs()`: 8 observations, drift, |x|, boulder dx/dy/r, waypoint dx,
  above-boulder, clean wind estimate. Computed only from past observations and the pilot's own past actions. Never
  `wind_idx`, `torque_idx`, the seed or future wind. Keep the test that recomputes every input from the observation
  history and flies the policy with `flight=None`.
- **Seed ranges** (separate, tested): train 0+, demos 5e8+, stage checks 1e9+, selection 1.5e9+, **goal test 2e9+**
  (block 1 = 2e9 + 0.., block 2 = 2e9 + 10,000..), baselines 2e9 + 50,000.., analysis 3e9+.

### The boulder, v3: every layout passable (approved by the user on 2026-10-31)

**Why:** in the old layout the boulder's top stayed ≥ 3.5 m below the start, but its bottom could sit only **1.73 m**
above the pad. The lander with its legs is about **1.8 m** tall and **1.5–2 m** wide. So when the boulder floated low
over the pad, the only way in was a sideways slide into a gap lower than the lander itself. The data agrees: on the last
project's 2,000 goal flights, the champion failed **2.5%** of layouts with the boulder low over the pad (gap < 2.5 m)
against **~1.15%** of all others.

**The rule (mirror the start rule at the pad):**

| Clearance | Old (v2) | **New (v3)** |
|---|---|---|
| Boulder top below the start height (13.33 m) | ≥ 3.5 m | **≥ 3.5 m** (unchanged) |
| Boulder bottom above the pad surface (3.33 m) | ≥ 1.73 m | **≥ 3.5 m** |
| Radius | 0.8–1.3 m | 0.8–1.3 m (unchanged) |
| Horizontal centre | pad centre ± 2.5 m | pad centre ± 2.5 m (unchanged) |

So the boulder's centre lies at `y ∈ [3.33 + 3.5 + r, 13.33 − 3.5 − r]` (a 3.0 m tall band, which fits the largest
boulder, 2.6 m across) and `x ∈ [7.5, 12.5]` m. It still sits on the approach (that is the point of the task), but
there is always ≥ 3.5 m of free air above it and below it, and ≥ 6.2 m of free air beside it on both sides
(the screen is 20 m wide). Implement it in `boulder_spec` with its own RNG stream, as before.

**Tests that prove it (write them first):**

1. **Geometry:** for 100,000 seeds of every range, the clearances above hold (top ≥ 3.5 m below the start, bottom
   ≥ 3.5 m above the pad, radius and x in range).
2. **Passability, demonstrated:** for every seed in **both goal blocks** and 2,000 analysis seeds, show that the layout
   can be flown. Use a privileged *planner* that is **only a test tool, never a policy**: it may read the true wind, and it
   tries a small set of routes (left of the boulder, right of it, over-then-down) with a simple controller, plus up to 20
   action-noise variants. A layout counts as passable if any try lands. **Required: 100% passable.** Write down the
   planner's own landed rate and any seed it needed many tries for. Show 3–4 of the hardest layouts on the site as replays.
3. **The old champion as a yardstick:** fly `../lunarlander-98/models/best/M2_i08100.json` (PPO, 98.6% on v2) on
   1,000 baseline seeds of v3. That is the number NEAT has to match. Re-measure the hand-written pilot v2 and the old
   NEAT seed clone the same way.

## 3. What the last project taught (use all of it)

- **Learning signal is the bottleneck for NEAT.** PPO won the bake-off (91.4% after 45 minutes vs 87.8% / 83.4%) because
  it learns from every step; NEAT, like ES, gets one number per flight per genome. So NEAT needs **many flights per
  genome, common random numbers** (all genomes of a generation fly the same seeds), **re-flying the top 8–10** on more
  selection flights, and **fast code** to afford all that (§6).
- **A confident copy is a cage.** PPO started from the behaviour clone barely moved (87.0% → 87.8%), and NEAT's
  mutations wrecked cloned networks in the project before (children averaged 70.5% from an 86% parent). Hence the
  **clean start**: random genomes, no seed network.
- **Hard cases gave the biggest single jump**: re-flying recently failed situations in 30% of training flights took PPO
  from a 92.7% plateau to ~96%, and boulder hits from 5% to ~2%.
- **Stage passes on 200 + 200 flights are passed by luck** after enough tries (24 failed confirmations before one
  "passed"). The final stage must **never end the run by itself**; archive snapshots and pick the champion on ≥ 1,000
  selection flights, re-flying the top few on 2,000 more.
- **Pay for landings:** 0.3 × game score + 300 per landing − 200 per timeout − 100 per boulder hit; no centre bonus.
  Watch the timeout rate, the off-pad rate and the touchdown speed at every check-in. (PPO ended with 0 off-pad landings
  in 2,000 flights, ~0.35% timeouts, touchdowns at speed ~0.26.)
- **Engineering traps we hit:** one BLAS thread per worker process (`OPENBLAS_NUM_THREADS=1`; 8 workers × 8 threads
  gave a load of 34); never `git pull --rebase --autostash` in the publisher (it silently reverted code for an hour);
  no commas in stage names, or read `stats.csv` with a CSV reader, never `awk -F,`; a finished curriculum must not leave
  the CPU idle (it did, for an hour).

## 4. The network: complex enough, from a clean start

Measure the needed size before evolving it: **distil the PPO champion into small MLPs** with DAgger (as a yardstick,
never as a seed) and fly each on 1,000 analysis seeds of v3:
`16→8→4`, `16→16→4`, `16→32→4`, `16→16→16→4`, optionally with 2–3 stacked frames. The smallest size that reaches ~98%
tells NEAT what it must be able to grow to. (A 16→24→4 clone reached the pilot's 85%; the PPO champion is 16→128→128→4,
which is surely more than needed.)

Then the NEAT genome:

- **Start:** all 16 inputs connected to a **hidden layer of H random tanh nodes** (H from the test above, e.g. 16) and
  to the 4 outputs, random weights (σ ≈ 1/√fan-in). Not a minimal genome: a minimal one spends days rediscovering what
  a hidden layer is for.
- **Mutation, gentle:** mostly small weight perturbations (σ ≈ 0.05–0.1 × weight scale, 80–90% of weights perturbed a
  little), weight *replacement* rare (≤ 1%), add-node / add-connection 2–5%, deletion very rare. Measure what one
  mutation does to a good genome (`experiments/mutation_damage.py` from `../lunarlander-curriculum`): children must not
  average more than ~3 points below their parent.
- Feed-forward first. Add recurrent connections (or stacked frames as inputs) only if the failure analysis shows the
  pilot cannot tell gusts apart from its own thrust.
- Population 200–400, speciation on, elitism 2–5, the champion re-flown on fresh seeds every generation.

## 5. A smart curriculum (shown on the site before each stage starts)

From a clean start, NEAT first has to learn to fly. Each stage has a world, a reward, a pass check, a generation cap
and an expected outcome, written on the Curriculum tab **before** it runs. Checks always fly the stage's final world
with no help, and every check also flies the **real task** (tracked on the chart).

1. **Calm landing:** no wind, no boulder, training wheels (an upright torque fading 1 → 0 over the stage).
2. **Wind ramp:** steady push → gusts 5 → 10 → 15, then turbulence 0 → 1.5, with rehearsal of calm flights.
3. **Boulder:** size 0.4 → 1.0 of full, the v3 layout, with **30–50% real-task flights** (rehearsal, so earlier
   skills are not forgotten).
4. **Real task + hard cases** (the long one, no early stop): 30% of each generation's flights re-fly
   **training-range** seeds the recent champions failed (keep it clean: no check or selection seeds in training);
   snapshot the champion every 25 generations.

Pass checks: 200 fresh check flights, confirmed on another fresh 200. Treat a pass only as "move on", never as proof.
If a stage plateaus (no gain over ~200 generations), change one thing, with the reason written down first.

## 6. Fast code: what we ended up using, and what NEAT needs

**What the last project used:** no native code. The reference gymnasium env in Python, the network forward pass in
numpy inside 8 worker processes (one BLAS thread each), PyTorch only for the PPO update. That gave **~18,000–20,000
steps/s** for PPO and ~40,000 steps/s for the hand-written pilot on this laptop (i5-8250U, 4 cores / 8 threads, 7.6 GB,
no usable GPU). It was enough because PPO needed only 254 M steps.

**NEAT needs much more.** Population 300 × 32 flights × ~300 steps ≈ 3 M steps per generation: about 2.5 minutes at
20,000 steps/s, so only ~500 generations a day. Target **≥ 10× faster** before the long runs:

- **Profile first** (the old project: Box2D itself 1.5 µs of 117 µs per step; the rest was Python glue).
- **Recommended: C++ with pybind11** (g++ and pybind11 are already on the machine; Rust/cargo is not installed). Compile
  **the same Box2D 2.3.x** that `box2d-py 2.3.5` wraps, port the LunarLander step logic (engines, legs, contact, wind,
  turbulence, the boulder, our boulder-hit and landed rules) and **evaluate the NEAT network in C++** too (genome →
  arrays in topological order), so one call flies a whole batch of flights with no Python in the loop.
- Generate each seed's start state in Python and pass it in (terrain heights, the initial random push, `wind_idx`,
  `torque_idx`, the boulder): gymnasium draws them from numpy's RNG, which is not worth porting. Cache them per seed.
- Rust via PyO3 is fine instead if you prefer it, but then the Box2D match is harder to get right.
- **The rule that keeps it honest:** the Python gymnasium env is the reference. The fast env may be used **for training
  only**, after a test shows that the same seeds and the same actions give the same outcome on ≥ 98% of 1,000 seeds, and
  the same agent's landed rate falls inside the reference's confidence interval. **All checks, the selection and the
  goal test run in the reference env.** Report speed-ups end to end (flights per second, generations per hour), before
  and after, on the Speed tab.

## 7. Reuse from `../lunarlander-98` (copy and adapt)

| What | File(s) |
|---|---|
| Task (frozen) + seed ranges | `lander/env.py`, `lander/route_params.json`, `lander/teacher.py`, `lander/seeds.py` |
| Policy loader, numpy MLP | `lander/policy.py` |
| Wind overlay for replays and films (true wind blue, pilot estimate orange, dust) | `lander/viz.py` |
| Evaluate / goal check (Wilson CI, both blocks) | `experiments/evaluate.py`, `experiments/goal_check.py` |
| Behaviour cloning + DAgger (for the size yardstick only) | `rl/bc.py` |
| The PPO champion (the yardstick to beat) | `models/best/M2_i08100.json`, `models/best/goal_*.json` |
| Run pieces: curriculum stages, checks, events, stats, snapshots, overrides, SIGTERM checkpoint | `rl/ppo.py` (`Run`), `supervise.sh` |
| Site + publisher (no autostash) + 15-min loop | `publish.py`, `publish_loop.sh`, `docs/` |
| Explainer film (scenes rendered from real flights + voiceover) | `video/build_film.py`, `video/bakeoff/script.md` |
| Tests | `tests/test_core.py` (frozen task, determinism, no-peeking, seed ranges, numpy/torch equivalence) |
| Old NEAT pieces: genome seeding, fast NEAT forward pass, mutation-damage test, gentle config | `../lunarlander-curriculum/lander/neatnet.py`, `train.py`, `experiments/mutation_damage.py`, `neat-config-gentle.ini` |
| Working method | `../lunarlander-curriculum/METHOD.md` |

## 8. The homepage and the podcast

- **Use the `ml-project-site` skill** (`~/.claude/skills/ml-project-site/`, `scripts/init.sh`): it sets up the whole site
  below, including the **Play tab** (the newest NEAT champions on a fixed, selectable set of games). Implement its
  `site_hooks.py` like `../lunarlander-98/site_hooks.py`. Clips and replays play once, never loop.

- New public repo (suggested `JCA374/lunarlander-neat98`; **ask the user before creating it**), Pages from
  `main:/docs`. Deployment counts only when `gh api repos/<repo>/pages/builds/latest` says `built <sha>` and the pages
  return HTTP 200.
- Tabs as last time: **Log** (newest first, every event with a replay that shows the wind; goal table; countdown;
  progress chart with autoscaled y-axis and fixed-height boxes), **Curriculum**, **How it sees**, **The model** (what NEAT
  is, how genomes grow, how the champion's network changed over time, with pictures of real genomes), **Speed**,
  **Lessons**, **Podcast**. Explain more rather than less.
- **Replays always draw the wind** (`lander/viz.py`) and the passability tool's hardest layouts get their own section.
- **A film** (like `video/build_film.py`) once the method is running: what NEAT is, the curriculum, the v3 layout and why
  it is fair, and the tests. A second film at the end.
- **Podcast for every major event** (skill `project-podcast`, `podcast.conf`): the plan, the v3 layout and passability
  result, the size yardstick, the fast env and its equivalence test, each curriculum stage passed, each plateau or trap,
  each correction, the final result. 2–6 minutes, first person, every number from the logs. No "still training" episodes.

## 9. How to run the work

1. **Day 1, in order:** ask for the deadline and the repo → measure the machine → repo, `.venv`, tests (determinism
   first) → v3 boulder + geometry test → passability test → baselines on v3 (PPO champion, pilot v2, old clone) → the
   size yardstick → site + episode 1 → a 1-hour NEAT smoke run in the Python env (does stage 1 learn at all?) → the fast
   env + equivalence test → the long run under the supervisor.
2. **Cron jobs** (session-only; set them up again after any restart): a check-in every 2 hours and the wrap-up one-shot.
   Check-in: look (processes, stats, memory, load), judge (improving, plateau, trap), decide (one change at a time, the
   reason written before the result), record (journal), publish, report a few lines in the chat. Nothing changed → one line.
3. **Verification:** pass checks on fresh check seeds with a confirmation on a second fresh set; champions chosen on
   ≥ 1,000 selection flights, top few re-flown on 2,000 more; "better" only from a paired comparison on ≥ 100 shared
   seeds with a bootstrap CI.
4. **Archive every champion**, never overwrite a verified best; `models/best/` with its goal-check JSONs.
5. **Correct mistakes in public**, in the journal and the next episode.
6. **Background jobs:** `setsid nohup`; SIGTERM must checkpoint; a supervisor restarts crashes; anchored `pgrep -f "^..."`
   patterns and kill by PID; kill test subprocesses by process group; narration and video with `nice`.
7. **Ask the user** only for things that change the task or go public (the repo, any further change to the world).

## 10. Final wrap-up (at the wrap-up time, or as soon as 98% is verified twice)

Stop the run cleanly (SIGTERM → checkpoint), the publish loop and the cron jobs. Goal check of the chosen champion on
test block 1 and block 2 in the reference env. Save it in `models/best/` with its results. Write `CONTINUE.md` (the
result with its CIs, exact resume commands, what worked, what didn't, open ideas), a final journal entry, a final podcast
episode, a last site update; commit, push, verify the Pages build. Then tell the user in the chat.

## 11. Lessons, in one list

- Learning signal decides speed: one number per flight needs many flights, shared seeds and fast code.
- Start clean; a copied pilot is a cage for RL and gets wrecked by NEAT's mutations.
- Measure the network size you need before evolving it.
- Hard cases are the curriculum for the last few percent.
- A stage pass on 400 flights is luck after enough tries; choose champions on thousands of fresh flights.
- Check that the world is fair, and prove it with a test (v3: passable by construction and by demonstration).
- Sensing beats network size: the clean wind sensor (0.97 correlation with the true wind) is why a small pilot can work.
- Pay for landings, not for the game score.
- Profile before rewriting; keep the reference env as the judge.
- Determinism, resume and git hygiene bugs cost the most time: test them first.
