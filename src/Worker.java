//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Worker
 * Class
 * 
 * Represents a network and some given configuration in a population.
 * 
 * @author Joshua Savoie
 */
public class Worker extends Thread {
    
    //-----[VARIABLES]-----\\

    protected volatile boolean run = false;
    protected volatile boolean waiting = false;
    protected boolean exploit = false;

    private final Database m_trainingData;
    private final Database m_testingData;

    private Network m_network;
    private Configuration m_config;

    private double m_testing;

    private double m_training;

    //-----[CONSTRUCTOR]-----\\
    /**
     * Worker
     * Constructor
     * 
     * Creates a worker in the population given some network and configuration.
     * 
     * @param network - The neural network to run.
     * @param config - The configurations to run the neural network on.
     * @param trainingData - The database to pull training data from.
     * @param testingData - The database to pull testing data from.
     */
    public Worker(Network network, Configuration config, Database trainingData, Database testingData) {

        m_network = network;
        m_config = config;
        m_trainingData = trainingData;
        m_testingData = testingData;

        m_testing = 0;

        m_training = 0;

    }

    //-----[METHODS]-----\\

    /**
     * run()
     * 
     * Thread executable.
     */
    public void run() {

        while(run) {

            for(int e = 1; e <= m_config.epochs; e++) {

                m_training = train();
                //m_testing = test();

                if(ready(e)) {
                    run = false;
                    waiting = true;
                    exploit = true;

                    break;
                }

            }

            while(waiting) {
                try {
                    this.sleep(50);
                } catch(Exception e) {
                    e.printStackTrace();
                    System.exit(0);
                }
            }

        }

    }

    /**
     * step()
     * 
     * Performs a mini-batch gradient descent step.
     * 
     * @return
     */
    private double train() {

        DataPoint[][] miniBatches = m_trainingData.generateMiniBatches(m_config.miniBatchSize);

        double tCost = 0;

        for(int b = 0; b < miniBatches.length; b++) {

            double[][] results = m_network.learn(miniBatches[b], m_config.learnRate);

            double mbCost = 0;

            for(int i = 0; i < results.length; i++) {

                double dpCost = 0;

                for(int r = 0; r < results[i].length; r++) {

                    dpCost += Math.pow(results[i][r] - miniBatches[b][i].y[r], 2);

                }

                mbCost += (dpCost / results[i].length);

            }

            tCost += (mbCost / results.length);

        }

        tCost = (tCost / miniBatches.length);

        return tCost;

    }

    /**
     * test()
     * 
     * Runs the neural network against the test data.
     * 
     * @return
     */
    private double test() {

        DataPoint[] data = m_testingData.getData();

        double score = 0;

        for(int dp = 0; dp < data.length; dp++) {

            double[] results = m_network.calculate(data[dp]);

            int d = 0;
            for(int r = 1; r < results.length; r++) {
                if(results[r] > results[d]) {
                    d = r;
                }
            }

            if(d == data[dp].label) {
                score++;
            }

        }

        return score / data.length;

    }

    private boolean ready(int epoch) {

        boolean ready = false;

        if(epoch == m_config.epochs) {
            ready = true;
        }

        return ready;

    }

    /**
     * getNetwork()
     * 
     * Returns a copy of the neural network.
     * 
     * @return
     */
    public Network getNetwork() {
        return m_network.clone();
    }

    /**
     * getConfiguration()
     * 
     * Returns a copy of the configurations.
     * 
     * @return
     */
    public Configuration getConfiguration() {
        return m_config.clone();
    }

    /**
     * getTestingScore()
     * 
     * @return
     */
    public double getTestingScore() {
        return m_testing;
    }

    /**
     * getTrainingScore()
     * 
     * @return
     */
    public double getTrainingScore() {
        return m_training;
    }

}
