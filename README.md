# Base Neural Network

---

The training of neural networks is centered in Simulation.java. This Simulation class utilizes PBT, or population-based training. PBT, in summary, acts as a hybrid approach merging the random search and hand-tuning methods. This is accomplished by having a population of neural networks (Population.java), each with random hyperparameters (mimicking the random search method), are run through a set of training data. From there, the highest scoring neural networks, known as the graduation class, are kept while the others are replaced with variations of the graduated neural networks.

---

## Instructions to Run

---

1. Run the program like normal. "CTRL + F5" or simply "F5."
2. Once the program is running, a prompt will begin.
    i. Answer each question exactly as how it prompts you to.
    ii. Note for population size: Every network receives its own thread. If you have 10 networks, you get 10 threads. Be careful not to memory overload.
3. Sit back and let it run.
    i. This can be stopped at any time.
    ii. Do note, while it is running, it will be saving files to whatever path you specified. It will, potentially, take up GBs worth of storage if you let it run for an extended period of time.
        a. This can be manipulated by changing the return String of Population.java's getGenerationFolder() method.

---

## Resources

---

Some notable videos that I watched to introduce me to and better understand the general concepts of neural networks are below:
- Gave me example problems to solve using the neural network. https://www.youtube.com/watch?v=hfMk-kjRv4c
- Provides a real neat abstraction using a piece of paper. https://www.youtube.com/watch?v=e5xKayCBOeU
- Helped with understanding the key concepts behind neural networks. https://www.youtube.com/watch?v=piF6D6CQxUw
- The math. https://www.youtube.com/watch?v=w8yWXqWQYmU
- Explains neural networks in really basic terms. https://www.youtube.com/watch?v=CqOfi41LfDw
- An amazing introduction into otherwise daunting concepts. https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- Population Based Training: https://www.youtube.com/watch?v=pEANQ8uau88&t=66

Notable Posts and Articles:
- Learning Rate: https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
- Weight Regularization: https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/
- Minibatch: https://scontent-msp1-1.xx.fbcdn.net/v/t39.8562-6/240818965_455586748763065_8609026679315857149_n.pdf?_nc_cat=111&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=_Tjlwa1D8wAAX-QKlkV&_nc_ht=scontent-msp1-1.xx&oh=00_AfAkfundMvNOw6Jty0KyMVtO8M1sxj8GoMo6kcc_89m2VQ&oe=638D1BA3
- Population Based Training: https://www.deepmind.com/blog/population-based-training-of-neural-networks

---

This project is licensed under the terms of the MIT License.
