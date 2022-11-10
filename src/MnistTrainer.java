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
     * start()
     * 
     * Starts the trainer thread. After this method is complete, the run() method will be automatically called.
     */
    public void start() {

    }

    /**
     * run()
     * 
     * The method that is run automatically after the start() method is called.
     */
    public void run() {

    }

}