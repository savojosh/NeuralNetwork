//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Trainer
 * Generic Trainer Class
 * 
 * Creates and manages a neural network's generations.
 * 
 * This class is meant to be extended where start() and run() are implemented by the user.
 */
public class Trainer implements Runnable {
    
    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private final int m_networkID;

    private String m_manifestFolder;
    private int m_generation;
    private Network m_network;

    //-----[CONSTRUCTOR]-----\\
    
    /**
     * Trainer
     * Generic Trainer Class
     * 
     * Creates and manages a neural network's generations.
     * 
     * This class is meant to be extended where start() and run() are implemented by the user.
     * 
     * @param manifestFolder = The location where the generations are saved to. Must be an absolute path.
     * @param networkID
     * @param numInputs
     * @param layerSizes
     * @param function
     */
    public Trainer(String manifestFolder, int networkID, int numInputs, int[] layerSizes, Layer.Functions function) {
        
        m_networkID = networkID;

        m_generation = 0;
        m_manifestFolder = manifestFolder;
        m_network = new Network(
            getGenerationFolder(),
            numInputs,
            layerSizes,
            function
        );

    }

    //-----[METHODS]-----\\

    /**
     * start()
     * 
     * Starts the trainer thread. After this method is complete, the run() method will be automatically called.
     * 
     * @Override this method and provide your own implementation.
     */
    public void start() {
        
    }

    /**
     * run()
     * 
     * The method that is run automatically after the start() method is called.
     * 
     * @Override this method and provide your own implementation.
     */
    public void run() {

    }

    /**
     * save()
     * 
     * Saves the network to a generation folder.
     */
    public void save() {

        m_network.setManifestFolder(getGenerationFolder());
        m_network.save();

        m_generation++;

    }

    public String getGenerationFolder() {
        return m_manifestFolder + "\\Generation_" + String.format("%03d", m_generation);
    }

}
