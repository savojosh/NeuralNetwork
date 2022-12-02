//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * MnistProcess
 * Class
 * 
 * An implementation of the Process class.
 * 
 * This implementation has the network run through the MNIST database and 
 * determine what digit is written in a 784 pixel picture.
 * The pixel values range from 0-255, 0 being white and 255 being black.
 * 
 * More on MNIST: http://yann.lecun.com/exdb/mnist/
 * 
 * @author Joshua Savoie
 */
public class MnistProcess extends Process {
    
    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private int m_id;

    private Network m_network;

    //-----[CONSTRUCTOR]-----\\
    /**
     * MnistProcess
     * Constructor
     * 
     * Creates a new MNIST task for a neural network.
     * 
     * @param network
     */
    public MnistProcess(Network network) {

        super(network);

        m_id = 0;
        m_network = network;

    }

    //-----[METHODS]-----\\

    /**
     * output()
     * 
     * Runs the neural network through the MNIST database and
     * has it determine what digit is shown in the picture.
     * 
     * The return is the average cost of neural network's outputs
     * across all the training dataset images.
     * 
     * @return
     */
    @Override
    public double output() {

        MnistDatabase database = MnistDatabase.getInstance();
        final int trainingSize = database.getTrainingDatasetSize();
        double[] costs = new double[trainingSize];

        // For each image in the training dataset...
        int score = 0;
        for(int i = 0; i < trainingSize; i++) {

            // Label is the digit in the 784 pixel picture.
            int label = database.getTrainingImageLabel(i);

            // Used to calculate cost.
            // Ex: Outputs [0,0,0,0,0,1,0,0,0,0] = Label 5
            double[] expectedOutputs = database.getTrainingImageExpectedOutputs(i);
            
            // Inputs into the neural network.
            int[][] matrix = database.getTrainingImageMatrix(i);
            double[] inputs = new double[matrix.length * matrix[0].length];

            // Formats the matrix from 2D to 1D.
            int in = 0;
            for(int r = 0; r < matrix.length; r++) {
                for(int c = 0; c < matrix[0].length; c++) {
                    inputs[in] = (double)matrix[r][c];
                    in++;
                }
            }

            // Have the network calculate a decision based on the inputs.
            double[] outputs = m_network.calculate(inputs);

            // The highest output is the network's decision.
            int decision = 0;
            for(int out = 0; out < outputs.length; out++) {
                if(outputs[out] > outputs[decision]) {
                    decision = out;
                }
            }

            // If the decision matched the label, score++.
            if(decision == label) {
                score++;
            }

            // Calculate the costs and store it into an array.
            costs[i] = m_network.cost(outputs, expectedOutputs);

        }

        // Saves the network's data to local files.
        m_network.save();

        // Calculates the average cost from all images.
        double avgCost = 0;
        for(double c : costs) {
            avgCost += c;
        }
        avgCost = avgCost / costs.length;

        System.out.println("Network " + m_id + " scored " + score + "/" + trainingSize + " with a cost of " + avgCost);

        return avgCost;

    }

    // TODO: Transition this into a constructor parameter.
    /**
     * setID()
     * 
     * @param id
     */
    public void setID(int id) {
        m_id = id;
    }

}
