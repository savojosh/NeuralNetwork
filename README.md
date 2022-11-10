# Base Neural Network

---

~~The training of neural networks (provided by the Trainer.java abstract class) is centered around population-based training (PBT). Population-based training, in summary, acts as a hybrid approach merging the random search and hand-tuning methods. This is accomplished by having a population of neural networks, each with random hyperparameters (mimicking the random search method), are run through a set of test data. From there, the highest scoring neural networks are kept while the others are replaced by clones of the highest scoring neural networks with variations added to them.~~
my own theory instead of this ^

I dumped four hours of refractoring my Trainer.java into a PBT system to a new method of training I wish to create/explore: Evolution Based Training.

This will involve having multiple networks generated and ran at parallel where they will auto update themselves. This method aims to combine the random-selection method and hand-tuning method (with Bayesian optimisation) to a hybrid approach where a set of networks are generated with random hyperparamters (like random-selection) but evolve over time (like hand-tuning).

In summary, a hybrid approach that is alternative to the population-based training method.

See this article on PBT: https://www.deepmind.com/blog/population-based-training-of-neural-networks

P.S. Some notable videos that I watched to introduce me to and better understand the general concepts of neural networks are below:
- Gave me example problems to solve using the neural network. https://www.youtube.com/watch?v=hfMk-kjRv4c
- Provides a real neat abstraction using a piece of paper. https://www.youtube.com/watch?v=e5xKayCBOeU
- Helped with understanding the key concepts behind neural networks. https://www.youtube.com/watch?v=piF6D6CQxUw
- The math. https://www.youtube.com/watch?v=w8yWXqWQYmU
- Explains neural networks in really basic terms. https://www.youtube.com/watch?v=CqOfi41LfDw

---

This project is licensed under the terms of the GNU General Public License v3.0.
