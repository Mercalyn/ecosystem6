Ecosystem 6 is a pytorch powered optimization script using an evolutionary algorithm. Neural networks written so that all population members for a given neural layer will matrix multiply simultaneously on the GPU. 
Independent neural network for each population member. Recommended population size 80-2000 depending on complexity of problem and size of networks.

Driver examples in /examples/ folder. 

Evolutionary methods:
- mutation (hardfork means to reroll a trait entirely, softfork to roll a nudge to existing trait)
- tournament (given bracket size, this slot's output will be replaced with highest scoring member)
- crossover (cross networks and ideas)
- reroll (totally reroll a population member, useful for introducing new genes consistently)
- 1-member-elitism (a single member who has the highest score gets to stay in the next simulation)
- stayover (like elitism in that a slot selected for stayover can stay into next simulation, but is randomized. helps lower selection pressure)

a note on elitism: multiple member elitism tends to be quite uniform in that the elites copy each other and tend to destroy the variability/diversity that gives unique solutions a chance.

version 6 (this version) includes some updates from previous, including a global config override which gives a one stop shop to be able to swap config files for testing/debugging/live. Tournament was rewritten to be much more efficient.
