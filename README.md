# Base Neural Network

---

The training of neural networks is centered in Simulation.java. This Simulation class utilizes PBT, or population-based training. PBT, in summary, acts as a hybrid approach merging the random search and hand-tuning methods. This is accomplished by having a population of neural networks (Population.java), each with random hyperparameters (mimicking the random search method), are run through a set of training data. From there, the highest scoring neural networks, known as the graduation class, are kept while the others are replaced with variations of the graduated neural networks.

P.S. Some notable videos that I watched to introduce me to and better understand the general concepts of neural networks are below:
- Gave me example problems to solve using the neural network. https://www.youtube.com/watch?v=hfMk-kjRv4c
- Provides a real neat abstraction using a piece of paper. https://www.youtube.com/watch?v=e5xKayCBOeU
- Helped with understanding the key concepts behind neural networks. https://www.youtube.com/watch?v=piF6D6CQxUw
- The math. https://www.youtube.com/watch?v=w8yWXqWQYmU
- Explains neural networks in really basic terms. https://www.youtube.com/watch?v=CqOfi41LfDw

---

This project is licensed under the terms of the MIT License.
