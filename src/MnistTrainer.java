//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * MnistTrainer
 * Class
 * 
 * An extension of the Trainer class to run the networks on the MNIST database.
 */
public class MnistTrainer extends Trainer {

    //-----[CONSTANTS]-----\\

    private static final int PERFECT_SCORE = 60000;
    private static final int CONSISTENCY_GOAL = 5;

    //-----[VARIABLES]-----\\

    private final int m_networkID;
    private final Network m_network;
    
    private int[] m_scores;

    //-----[CONSTRUCTOR]-----\\
    /**
     * MnistTrainer
     * Class
     * 
     * An extension of the Trainer class to run the networks on the MNIST database.
     * 
     * @param manifestFolder
     * @param networkID
     * @param numInputs
     * @param layerSizes
     * @param function
     */
    public MnistTrainer(String manifestFolder, int networkID, int numInputs, int[] layerSizes, Layer.Functions function) {

        super(manifestFolder, networkID, numInputs, layerSizes, function);

        m_networkID = networkID;

        m_network = super.getNetwork();

    }

    //-----[METHODS]-----\\

    /**
     * run()
     * 
     * The method that is run automatically after the start() method is called.
     */
    public void run() {

        // Gets the MNIST database.
        MnistDatabase database = MnistDatabase.getInstance();

        // The number of times the network scored perfectly.
        int consistency = 0;

        // While the thread isn't stopped.
        while(!isStopped()) {

            // The network's score.
            int score = 0;

            // Loops through each image in the training dataset.
            for(int i = 0; i < database.getTrainingDatasetSize(); i++) {

                int label = database.getTrainingImageLabel(i);
                int[][] matrix = database.getTrainingImageMatrix(i);
                double[] inputs = new double[matrix.length * matrix[0].length];

                // Formats the matrix into a 1D array of inputs.
                int in = 0;
                for(int r = 0; r < matrix.length; r++) {
                    for(int c = 0; c < matrix[0].length; c++) {
                        inputs[in] = (double)matrix[r][c];
                        in++;
                    }
                }

                try {
                    // Calculates the networkss outputs.
                    double[] outputs = m_network.calculate(inputs);

                    // Determines the network's decision by finding which output had the greatest value.
                    int decision = 0;
                    for(int out = 0; out < outputs.length; out++) {
                        if(outputs[out] > outputs[decision]) {
                            decision = out;
                        }
                    }

                    // If the decision correctly matched the label, up its score.
                    if(decision == label) {
                        score++;
                    }

                } catch (Exception e) {
                    e.printStackTrace();
                }

            }

            System.out.println("Network " + m_networkID + "." + getEvolution() + " scored " + score + "/" + PERFECT_SCORE);

            // Saves the current network to local.
            save();

            // If it scored perfectly, increase the consistency rating.
            // If it did not score perfectly, have the network evolve.
            if(PERFECT_SCORE == score) {
                // save();
                consistency++;
            } else {
                m_network.update(score, PERFECT_SCORE);
                // incrementEvolution();
            }

            // If the network reached the consistency goal, end the thread.
            if(consistency == CONSISTENCY_GOAL) {
                // save();
                end();
            }

        }

    }

}