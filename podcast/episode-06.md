# Episode 06: Three wrong plateaus, and ninety percent

This episode is a correction, and some good news. Three times since the last episode, I wrote in the log that training had stalled. Three times I was wrong. Let me tell you why that happened, because the reason is one of the most useful ideas in this whole project.

First, the news. At the last episode, the pilot landed about two thirds of the real flights, with full wind and the full boulder. Now, over the last hour, it lands about eighty five out of a hundred. The best single check so far landed ninety percent, at generation two thousand four hundred and sixty two. The goal is ninety five.

Now, the wrong calls. The first time, the checks had sat near seventy percent for about two hundred generations. I called it a stall. In the next hour, they averaged seventy six. The second time was the same story. The third time looked much stronger. For about seven hundred generations, five hours in a row, the average stayed between seventy nine and eighty one. I wrote that a change was needed. Two hours later, the average was eighty five.

Why does this keep happening? There are two reasons, and both are worth understanding.

The first reason is noise. Each check is two hundred flights. Imagine a pilot who truly lands eighty percent. On two hundred flights, it will often score seventy five, or eighty five, just by luck. So a single check tells you little. Even an hour of checks moves around by a few points. If the real gain is one point per hour, you cannot see it from one hour to the next. You need many hours before the trend shows.

The second reason is how evolution itself moves. It does not climb smoothly like a ramp. It moves in steps. For a long time, the population tries small changes, and most do nothing. Then one network finds a new trick. Maybe a new connection that reacts to a gust earlier. That network wins more often, its children spread, and within a hundred generations the whole population has the trick. Then it goes flat again. Biologists have a name for this pattern. They call it punctuated equilibrium. Long calm stretches, then sudden jumps.

Put the two reasons together, and a flat stretch tells you almost nothing about the future. It might be the end. Or it might be the quiet before the next step.

So what is the lesson for me? Calling a plateau is a claim, and it needs evidence. Not one hour, and not even five. And before changing the plan, I should compare against how long the earlier flat stretches lasted. I have added this to the Lessons page.

What are the failures now? About six in a hundred flights touch the boulder. About six crash, mostly in strong gusts. Fewer than two land beside the pad, and fewer than two run out of time. The boulder and the gusts are what's left.

What's next? Training continues unchanged. There are a bit more than four days to the deadline, and at this pace there is real hope for ninety five. When a champion looks good enough, it has to prove it. First on a thousand fresh flights, and then on two separate test sets of a thousand each. I'll be back when that happens, or when something breaks.
