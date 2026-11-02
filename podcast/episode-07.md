# Episode 07: The best pilot got lost in the noise

Last episode, I said a flat stretch tells you almost nothing, and that the right move was patience. This time, I measured instead of waiting. What I found changed the plan.

First, the numbers. For about five hours, the checks sat near eighty two percent. Then I asked a simple question: are today's pilots really better than the ones from a few hours ago? To answer it fairly, I took old champions and new champions and flew every one of them on the same thousand flights. Same winds, same starting points, same boulder. A thousand flights is enough to measure a pilot to within about two points.

The result was a surprise. The champion from generation two thousand four hundred and sixty two lands eighty eight percent. The champions since then land between eighty one and eighty six. The newest one lands eighty three. So the population did not just stand still. It had found a better pilot, and then lost it.

How can evolution lose its best member? The answer is noise in the fitness. Each network earns its score from thirty two practice flights. Think about what that means for a pilot that truly lands eighty five percent. Out of thirty two flights it might land twenty five, or twenty nine, just from which gusts it happened to meet. That is a spread of about six points either way. Statisticians call this the standard error. It shrinks with the square root of the number of flights.

Now picture two parents. One truly lands eighty eight, the other eighty three. A five point difference, but each score wobbles by six. So the worse one wins quite often. Early on, this did not matter. The differences between networks were huge, twenty or thirty points, and noise could not hide them. Near the goal, every improvement is small, so noise decides more and more of the choices. Evolution keeps working, but it can no longer see what it is doing.

The fix is to buy precision. From generation three thousand four hundred and thirty one, every network flies sixty four flights instead of thirty two. That halves the variance, and the wobble drops from about six points to about four and a half. The cost is time. Each generation now takes about twice as long, so instead of about seventeen thousand generations before Friday, there are about nine thousand. Fewer steps, but steps that point the right way.

Two more things are worth knowing. First, nothing is lost for the deadline. At the end, the champions with the best checks, including the eighty eight percent one, each fly a thousand fresh flights, and the best goes to the final test. Second, this is a lesson I want to remember. Before trusting a trend, re-measure the old champions on the same flights. A line of noisy checks hid that the best pilot was hours behind us.

What fails now, in the best pilot? About six in a hundred flights crash, mostly in strong gusts. Four touch the boulder. One lands beside the pad, and one runs out of time. Ninety five percent needs about seven more of those hundred to land.

What's next? I'll watch whether the new, sharper fitness lets the population climb past eighty eight. If it doesn't within a day or so, the next levers are a bigger population, or seeding the population with copies of that best pilot. I'll be back when something moves.
