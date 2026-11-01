# Episode 03: Five times faster, and why not ten

This episode is about speed. In evolution, speed is everything. Every generation, three hundred networks each fly twenty four flights. That is about two million small physics steps. The faster we fly, the more generations fit before Friday.

This morning, one generation took about a hundred seconds. Now it takes about nineteen. That is a bit more than five times faster. Here is how, and what it teaches.

The first rule of speed work is to measure before you change anything. Where does the time go? I timed every part of one flight step. Some went to the physics. A lot went to Python itself. Python is easy to write, but every small operation has overhead. Our simulator wraps the physics in several layers, and each layer costs a little on every step.

Step one was the network. Every step, the network reads sixteen numbers and picks one of four actions. I compiled that part to machine code with a tool called numba. One decision went from forty six millionths of a second to five. Step two was a lean copy of the flight code, without the layers. Together, a generation went from ninety five seconds of flying to forty eight.

Step three was the big one. I rewrote the whole flight in C plus plus. It uses the very same physics engine, compiled from the same source code. One call now flies a whole batch of flights with no Python in the loop. Flying time dropped to eighteen seconds.

But a faster simulator is only useful if it is the same simulator. If it differs, evolution learns the wrong world. So I tested it. The same network flew the same thousand flights in both versions, in three different worlds. Three thousand of three thousand flights matched bit for bit, every position and every number. Even so, the rules stay strict. The fast version is only for training. Every check, every choice of champion and the final test still fly the original.

That test found something surprising. The original simulator does not always agree with itself. If you reuse it for many flights, the last digits can differ after touchdown. The physics engine recycles the internal numbers of deleted objects, and that changes the order of tiny sums. It almost never changes an outcome, but it is a good reminder. Exact repeatability is something you have to test, not assume.

Then a new surprise. With flying at eighteen seconds, a generation still took twenty seven. The other nine went to the evolution library itself, while all workers sat idle. Making children was slow. For every gene a child inherits, the library checked the whole network for loops, again and again. I rewrote that check and the distance between networks with the same logic. Same children, same species, four and a half times faster. This is a classic pattern. Speed up the biggest part, and the second biggest part becomes the new wall.

Last, the number of workers. The laptop has four real cores, and each can run two threads. I measured four, five, six, seven and eight workers on the same generations. Eight was best, but only seventeen percent faster than four. Two threads on one core share the same machinery, so the second thread adds little.

Why not ten times, as planned? Because the physics engine itself is the floor. This game asks it to solve every step with a hundred and eighty passes for speed and sixty for position. That work is the same in any language. Making it cheaper would mean changing the physics, and then we would be learning a different game. So we stop at five.
