//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Trainer
 * Abstract Trainer Class
 * 
 * Creates and manages a neural network's evolutions.
 * 
 * This class is meant to be extended where start() and run() are implemented by the user.
 */
public abstract class Trainer implements Runnable {
    
    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private final int m_networkID;
    private final Network m_network;

    private String m_manifestFolder;
    private int m_evolution;

    //-----[CONSTRUCTOR]-----\\
    
    /**
     * Trainer
     * Abstract Trainer Class
     * 
     * Creates and manages a neural network's evolutions.
     * 
     * This class is meant to be extended where start() and run() are implemented by the user.
     * 
     * @param manifestFolder = The location where the evolutions are saved to. Must be an absolute path.
     * @param networkID
     * @param numInputs
     * @param layerSizes
     * @param function
     */
    public Trainer(String manifestFolder, int networkID, int numInputs, int[] layerSizes, Layer.Functions function) {
        
        m_networkID = networkID;

        m_evolution = 0;
        m_manifestFolder = manifestFolder;
        m_network = new Network(
            getEvolutionFolder(),
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
     */
    public abstract void start();

    /**
     * run()
     * 
     * The method that is run automatically after the start() method is called.
     */
    public abstract void run();

    /**
     * save()
     * 
     * Saves the network to a evolution folder.
     */
    public void save() {

        m_network.setManifestFolder(getEvolutionFolder());
        m_network.save();

        m_evolution++;

    }

    /**
     * getEvolutionFolder()
     * 
     * @return
     */
    public String getEvolutionFolder() {
        return m_manifestFolder + "\\Evolution_" + String.format("%03d", m_evolution) + "\\Network_" + String.format("%03d", m_networkID);
    }

    /**
     * getNetwork()
     * 
     * @return
     */
    protected Network getNetwork() {
        return m_network;
    }

}
