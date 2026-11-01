# Episode 05: Practising the hard cases, and a plateau

This episode covers the last stretch of the night. The boulder stage ended, the final stage began, and the climb slowed down a lot. Let me explain what happened, and then what it means.

The boulder stage had a simple plan. The rock grows from small to full size as the champion gets better. It reached full size at generation five hundred and seventy eight. But the stage also had a test. It needed eighty five percent on the real task to pass. It never got there. So the stage ended at its time limit, at generation seven hundred and two. Over its last fifty generations, the real task was at about sixty five percent.

Then the final stage began. This is the real task itself, with full wind, full gusts and the full boulder. It adds one new idea, called hard cases.

Here is the idea. Most flights are easy for a good pilot. If every flight counts the same, the easy ones drown out the few that matter. So we keep a list of flights that recent champions failed. About three in ten flights in each generation are taken from that list. A network that learns to handle them gets a clear advantage. Teachers do the same thing. You practise the problems you got wrong, not the ones you already know. In the last project this trick lifted the pilot from about ninety three percent to about ninety six.

So, did it work here? A little. The real task is checked every ten generations, on two hundred fresh flights. In the first sixty generations of this stage, those checks averaged sixty four percent. In the next seventy, sixty seven. The best single check was seventy three and a half percent. The best ever is still seventy six, from much earlier, and that one was probably a lucky draw.

What goes wrong in the failed flights? About thirteen in a hundred are crashes. About ten touch the boulder. About eight land safely, but beside the pad. And three or four run out of time. So it is not one problem. It is several smaller ones, and most of them come from the strong gusts.

Now the honest part. The goal is ninety five percent by Friday. At this pace, about three points per hundred generations, we will not get there. This is what a plateau looks like. Evolution keeps the best networks, but each small improvement is harder to find than the one before. And with only two hundred flights per check, a real gain of two points is hidden in the noise.

There is a second, quieter cause. Each network flies thirty two flights per generation. Two networks of equal skill can differ by several flights just by luck. Evolution then picks the lucky one, not the better one. With more flights per network, or more networks, selection gets cleaner, but each generation costs more time. That is the trade-off at the heart of this method.

So what's next? There are a few levers. More flights per network, so selection is less noisy. A larger share of hard cases. A reward that punishes the specific failures more, like landing beside the pad. Or a bigger population, so there are more ideas to choose from. I'll propose one to you, measured first, rather than guess. Training keeps running in the meantime.
