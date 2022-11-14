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

    private volatile boolean stopped;

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

        stopped = false;

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
     * run()
     * 
     * The method that is run automatically after the start() method is called.
     */
    @Override
    public abstract void run();

    /**
     * end()
     * 
     * Ends the thread.
     */
    public void end() {
        stopped = true;
    }

    /**
     * isStopped()
     * 
     * @return
     */
    public boolean isStopped() {
        return stopped;
    }

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
     * getEvolution()
     * 
     * @return
     */
    public int getEvolution() {
        return m_evolution;
    }

    public void incrementEvolution() {
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
