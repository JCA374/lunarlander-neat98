# What PPO does that NEAT does not

## Two pilots

These are two pilots flying the same kind of hard task. Strong gusty wind, turbulence, and a boulder on the way down. On the left is PPO, from the last project. It lands about ninety nine flights in a hundred. On the right is NEAT, from this project. Its best pilot lands about ninety three. They learned in completely different ways. This film is about what PPO does that NEAT does not do, and why that matters.

## The scoreboard

First, the history. Before the PPO project, there were two NEAT projects, and both topped out at about eighty five percent. Then PPO reached ninety eight point eight and ninety eight point four percent, on two separate test sets of a thousand flights. This project is NEAT again, done carefully, and its best pilot is at ninety three. Now look at the cost. PPO needed about two hundred and fifty million simulated steps, roughly four and a half hours of training. NEAT has used more than thirty billion steps so far. That is over a hundred times more experience, for a worse pilot. The rest of this film explains where that difference comes from.

## How NEAT learns

Start with NEAT. It keeps a population of three hundred networks. Each one flies the same set of flights, and at the end it gets one number, its fitness. That number mixes how often it landed with the game's score. The best networks become parents. Their children are copies with small random changes: a weight nudged, a new connection, sometimes a new neuron. Then the next generation flies, and so on. Notice what NEAT never looks at. It never asks which moment of a flight went well or badly. A whole flight, hundreds of decisions, is boiled down into one number, and that number is all evolution gets to see.

## Every step counts

PPO works differently. While it flies, it writes down every single step. What it saw, which engine it fired, how sure it was about that choice, and the reward right after it. In this flight, that is two hundred and four steps, and each one becomes a lesson. And PPO does not just pick the best action. It keeps a probability for each of the four choices. Here you can see them change as it flies: do nothing, left engine, main engine, right engine. During training it rolls the dice with these probabilities, so it keeps trying slightly different things. When it is tested, it simply takes the most likely action.

## The critic

To learn from single steps, PPO needs to know which steps were good. For that it trains a second network, called the critic. The critic looks at the situation and predicts how many points the rest of the flight will bring. Here is the real critic from the last project, along this flight. At the start, it expects about a hundred and seventy points. As the lander gets closer to a safe landing, the prediction rises toward three hundred. The critic is like a coach who knows from experience how promising each moment is.

## Better or worse than expected

Now the key idea, called the advantage. After each step, PPO compares what actually happened with what the critic expected. If the step led somewhere better than predicted, the advantage is positive, and that action should become more likely. If it led somewhere worse, the advantage is negative, and that action should become less likely. The green bars are steps that went better than expected, and the red bars are steps that went worse. The method that does this adding up is called generalized advantage estimation. It looks ahead, counting the nearest steps the most, and blends what the rewards actually were with what the critic predicted. Two numbers control the blend. In the last project they were zero point nine nine five and zero point nine five. This is how PPO answers the question NEAT never asks: which decision, in which moment, made the difference.

## Gradients: every weight gets a direction

With an advantage for every step, PPO can use calculus. Backpropagation works out, for each of the nineteen thousand weights in the pilot's network, which direction would make good actions more likely and bad actions less likely. Then every weight moves a tiny bit in its own best direction, all at once. NEAT has no such map. It changes weights at random and keeps the changes that happen to score well. Imagine finding the bottom of a valley. PPO feels the slope under its feet. NEAT throws darts and walks to whichever one landed lowest.

## Small, careful steps

There is a danger in following the slope too eagerly. One big step can wreck a pilot that was almost good. PPO's main trick is to stop that. For each step of experience, it compares the new pilot's probability of the action with the old pilot's. That is a ratio. If the ratio moves outside a small window, by default twenty percent either way, PPO stops pushing it further. That is the clip in the name: proximal means staying close. Because the steps are kept small, PPO can safely reuse the same batch of experience several times. In the last project, each batch of about sixteen thousand steps was used four times, and an update also stopped early if the pilot started to change too much. NEAT uses each flight once and throws it away.

## Exploring

Both methods have to explore, but in different places. PPO explores in what it does. The dice it rolls on every step try slightly different actions, and a small bonus in its goal, called the entropy bonus, rewards it for not becoming too certain too early. NEAT explores in what it is. Each child is a slightly different network, and the whole network flies a whole flight before anyone learns anything.

## What NEAT has that PPO does not

That does not make NEAT useless. It has strengths PPO lacks. It grows its own structure. Our best NEAT pilot has forty one hidden neurons and a hundred and ten connections, more than a hundred times fewer weights than PPO's. It needs no gradients, no critic, and no rule about rewards per step, so it works even when the only thing you can measure is the final result. And every network can fly on its own processor, with nothing to share between them. Species protect brand new ideas long enough to prove themselves. Researchers have shown that evolution methods can compete with reinforcement learning when they have lots of computers. On one laptop, the price is the one number per flight.

## Why this matters here

That price is exactly what we keep running into. Because NEAT only sees one noisy number per network, it needs many flights to tell a good pilot from a lucky one. This project went from thirty two flights per network to sixty four, then to a hundred and twenty eight, each time because the noise was hiding the progress. PPO faces noise too, but it fights it with thousands of steps per update and a critic that removes much of the luck. So the short answer is this. PPO learns from every moment, knows which moments mattered, moves every weight in a useful direction, and takes careful steps. NEAT learns from whole flights, by trial and selection. Both can land a lander in a storm. One of them just needs a lot more storms.
