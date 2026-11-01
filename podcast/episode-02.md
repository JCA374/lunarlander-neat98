# Episode 02: Species, wind, and a correction

Welcome back. In the first episode, evolution learned to land in calm air. This one covers the wind stage. It also covers a mistake I told you about wrongly last time, so let me start there.

First, a quick idea you need. In this method, every network belongs to a species. A species is a group of networks that are built alike. Networks only compete hard inside their own species. Why does that matter? A new neuron usually makes a network worse at first. Its weights are still random. Without protection, that network dies before the new neuron becomes useful. Species give new ideas a few generations of shelter. So the number of species is a health sign. Too few, and new ideas die. Too many, and each group is too small to improve.

To decide who belongs together, the method measures a distance between two networks. It counts the genes they do not share, and how different the shared weights are. If the distance is below a threshold, they share a species.

In the last episode I said the species had collapsed to three because my lower limit on the threshold was too high. That was wrong. I lowered the limit and nothing changed. So I measured instead of guessing. Members of each species sat far from each other. Far more than the threshold allowed. That was impossible, unless the threshold in use was not the one I was setting. And that was the bug. When the run restarted from a saved file, the part that sorts species loaded its own old copy of the settings. It kept a runaway value of forty six from an early restart. My controller had been tuning a number that nobody read. For a hundred and forty generations, everything was lumped into old species until only three were left.

The lesson is general. When you save and resume a program, check that the resumed program really uses the live settings. Now a test checks exactly that.

The fix caused one more problem. With the real threshold back, my first choice split the population into a hundred and five species. Each species needs at least three members. A hundred and five times three is more than three hundred, so the program stopped. This time I chose by measurement. I sorted the saved population with different thresholds and counted the species. Zero point four gave thirteen. That is what the run uses now.

Now the wind. The wind grew slowly, from a soft steady push to the full gusty wind with turbulence. At generation two hundred and eleven I checked the champion in different worlds. In its own practice world, it landed ninety seven and a half percent. In full wind, sixty seven percent. That was up from fifteen percent at the start of the stage. So wind learning worked. But with the boulder, even in calm air, it landed zero percent. It had simply never met the boulder.

That check also showed a second problem. The wind grew on a fixed clock. The champion was landing almost every flight, and still had to wait. A good teacher does not wait when the student is bored. So I made the ramp follow the student. When the champion lands nine flights out of ten, the wind grows three times faster. Between seven and nine, normal speed. Below seven, half speed. This idea is called pacing. You keep the learner at the edge of what it can do. Not so easy that time is wasted, and not so hard that it fails every time.

I also mixed a small boulder into a quarter of the wind flights, so the route skill could start early.

It worked. At generation three hundred and one, the wind stage passed. The champion landed eighty seven and a half percent of two hundred fresh flights in full wind, with no boulder. Then the boulder stage began. That is the next story, together with a big speed-up.
