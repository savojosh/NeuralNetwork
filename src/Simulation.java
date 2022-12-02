//-----[IMPORTS]-----\\

import java.util.concurrent.TimeUnit;

//-----[CLASS]-----\\
/**
 * Simulation
 * Thread Manager Class
 * 
 * Creates and manages threads to train a population of networks.
 * 
 * @author Joshua Savoie
 */
public class Simulation {
    
    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private int m_graduationSize;
    private Population m_population;

    private final Class<? extends Process> m_task;

    //-----[CONSTRUCTOR]-----\\
    /**
     * Simulation
     * Constructor
     * 
     * Creates a new simulation.
     * 
     * @param graduationSize = How many neural networks qualify for the next generation.
     * @param population = The population of neural networks.
     * @param task = The task for the neural networks to complete.
     */
    public Simulation(int graduationSize, Population population, Class<? extends Process> task) {

        m_graduationSize = graduationSize;
        m_population = population;

        m_task = task;

    }

    //-----[METHODS]-----\\

    /**
     * run()
     * 
     * Creates threads for each neural network to run them simultaneously.
     * Threads will automatically terminate.
     */
    public void run() {

        // Infinite loop.
        /* TODO:
         * Change the while(true) to some sort of function for when
         * the neural networks achieve perfection, stop?
         */
        do {

            // Gets the neural networks and prepares to store their costs.
            Network[] networks = m_population.getNetworks();
            double[] costs = new double[networks.length];

            // "completed" prevents multiple printlines for one thread.
            boolean[] completed = new boolean[networks.length];
            // Loops through the boolean[] and sets all values to false.
            for(int c = 0; c < completed.length; c++) completed[c] = false;

            SimulationThread[] threads = new SimulationThread[m_population.getSize()];
            // Threads active tracks how many threads are currently active.
            // This variable implements loose pseudo-synchronization for the threads.
            int threadsActive = 0;

            // Creates the threads and starts them.
            for(int t = 0; t < threads.length; t++) {
                threads[t] = new SimulationThread(networks[t], m_task);
                threads[t].setID(t);
                threads[t].start();

                threadsActive++;
            }

            // While the number of threads active is > 0.
            while(threadsActive > 0) {

                // Check through all of the threads to see if they are active.
                for(int t = 0; t < threads.length; t++) {
                    // If they are no longer active...
                    // store the neural network's cost;
                    // set completed[thread] to true;
                    // and decrement threadsActive.
                    if(!threads[t].isAlive() && !completed[t]) {
                        costs[t] = threads[t].getCost();
                        // System.out.println("Network " + t + " has finished with a cost of " + costs[t]);

                        completed[t] = true;
                        threadsActive--;
                    }
                }


            }

            System.out.println("Generation " + m_population.getGeneration() + " is now learning...");
            
            // Repopulates the population according to the graduation size and the neural networks' costs.
            m_population.repopulate(m_graduationSize, costs);

            System.out.println("Generation " + m_population.getGeneration() + " is now ready.");

        } while(true);

    }

}
