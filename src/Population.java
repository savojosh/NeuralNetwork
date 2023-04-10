//-----[IMPORTS]-----\\

import java.util.ArrayList;

//-----[CLASS]-----\\
/**
 * Population
 * Class
 * 
 * Represents a population of neural networks.
 * Based on the Population-Based Training algorithm.
 * developed by Google Deepmind.
 */
public class Population extends Thread {

    //-----[VARIABLES]-----\\

    protected boolean run = false;

    private final Database m_trainingData;
    private final Database m_testingData;
    
    private Worker[] m_workers;
    private ArrayList<Integer> m_miniBatchSizes;

    //-----[CONSTRUCTOR]-----\\
    /**
     * Population
     * Constructor
     * 
     * Creates a new population of neural networks.
     * 
     * @param populationSize
     * @param layerSizes
     * @param trainingData
     * @param testingData
     */
    public Population(int populationSize, int[] layerSizes, Database trainingData, Database testingData) {

        m_trainingData = trainingData;
        m_testingData = testingData;

        // Calculates the possible all of the possible mini-batch sizes that still take each training image
        m_miniBatchSizes = new ArrayList<Integer>();
        int dataSize = m_trainingData.getData().length;
        for(int n = 1; n < dataSize; n++) {
            if(dataSize % n == 0) {
                m_miniBatchSizes.add(n);
            }
        }

        // Creates all of the neural network populations.
        m_workers = new Worker[populationSize];
        for(int w = 0; w < m_workers.length; w++) {

            /*
             * Best to keep epochs 10+.
             * It will take longer for workers to exploit, but it prevents progress from being reversed
             * as it gives ample time for the workers to learn.
             */
            Configuration config = new Configuration(
                15,
                m_miniBatchSizes.get((int)(Math.random() * (m_miniBatchSizes.size() - 15)) + 15), 
                (Math.random() * 0.39 + 0.01)
            );

            Network network = new Network(
                "", 
                784, 
                layerSizes
            );

            m_workers[w] = new Worker(network, config, m_trainingData, m_testingData);

        }

    }

    /**
     * run()
     * 
     * Thread executable.
     */
    public void run() {

        // Starts up all of the threads.
        run = true;

        for(int w = 0; w < m_workers.length; w++) {
            m_workers[w].run = true;
            m_workers[w].start();
        }

        // Run process
        while(run) {

            // Checks through each of the workers if they are ready to exploit another worker
            for(int w = 0; w < m_workers.length; w++) {

                if(m_workers[w].exploit) {

                    // Determines which worker to exploit
                    int toExploit = w;
                    for(int e = 0; e < m_workers.length; e++) {

                        if(m_workers[toExploit].getTrainingScore() > m_workers[e].getTrainingScore() && m_workers[e].getTrainingScore() != 0) {
                            toExploit = e;
                        }
                    }

                    // Ensures that the worker isn't exploiting themself
                    if(w != toExploit) {

                        Configuration oc = m_workers[toExploit].getConfiguration();

                        // Mini-batch size
                        int fPos = (int)(m_miniBatchSizes.size() / 2);
                        for(int f = 0; f < m_miniBatchSizes.size(); f++) {
                            if(m_miniBatchSizes.get(f) == oc.miniBatchSize) {
                                fPos = f;
                                break;
                            }
                        }
                        // Exploration of sizes
                        double randomVal = Math.random();
                        if(randomVal > 0.9 && fPos < m_miniBatchSizes.size() - 2) {
                            fPos += 2;
                        } else if(randomVal > 0.55 && fPos < m_miniBatchSizes.size() - 2) { 
                            fPos += 1;
                        } else if(randomVal < 0.1 && fPos > 1) {
                            fPos -= 2;
                        } else if(randomVal < 0.45 && fPos > 0) {
                            fPos -= 1;
                        }

                        // Exploration of learn rates
                        double learnRate = oc.learnRate;
                        randomVal = Math.random();
                        if(randomVal > 0.6 && learnRate < 0.714285) {
                            learnRate *= (1.2 + ((Math.random() * 0.4) - 0.2));
                        } else if(randomVal < 0.4) {
                            learnRate *= (0.8 + ((Math.random() * 0.4) - 0.2));
                        }

                        // Exploration of epochs (just determines how long before the worker is ready to exploit)
                        int epochs = m_workers[w].getConfiguration().epochs;
                        // randomVal = Math.random();
                        // if(randomVal > 0.55 || epochs < 3) {
                        //     epochs += 1;
                        // } else if(randomVal < 0.5) {
                        //     epochs -= 1;
                        // }

                        // Creates the new configuration given those explorative values
                        Configuration config = new Configuration(
                            epochs, 
                            m_miniBatchSizes.get(fPos), 
                            learnRate
                        );

                        // Terminates the worker thread
                        m_workers[w].waiting = false;

                        m_workers[w] = new Worker(
                            m_workers[toExploit].getNetwork(),
                            config,
                            m_trainingData,
                            m_testingData
                        );

                        // Starts the worker
                        m_workers[w].run = true;
                        m_workers[w].start();

                    } else {

                        // Tells the worker to run again
                        m_workers[w].exploit = false;
                        m_workers[w].run = true;
                        m_workers[w].waiting = false;

                    }

                }

            }

            // Clears the console
            try {
                this.sleep(200);
                new ProcessBuilder("cmd", "/c", "cls").inheritIO().start().waitFor();
            } catch(Exception e) {
                e.printStackTrace();
            }

            // Prints out each of the worker's progress and hyperparameters
            for(int w = 0; w < m_workers.length; w++) {
                Configuration config = m_workers[w].getConfiguration();
                System.out.printf("Worker " + (w + 1) + ":\t");
                System.out.printf("Training Data = %12.10f\t ", m_workers[w].getTrainingScore());
                //System.out.printf("Testing Data = %6.4f\t ", m_workers[w].getTestingScore());
                System.out.printf("Epochs = %d\t ", config.epochs);
                System.out.printf("Mini-Batch Size = %d\t ", config.miniBatchSize);
                System.out.printf("Learning Rate = %6.4f%n", config.learnRate);
            }

        }

        for(int w = 0; w < m_workers.length; w++) {

            m_workers[w].run = false;
            m_workers[w].exploit = false;
            m_workers[w].waiting = false;

        }

    }

}
