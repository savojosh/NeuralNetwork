//-----[IMPORTS]-----\\

import java.util.Arrays;
import java.io.File;
import java.util.ArrayList;

//-----[CLASS]-----\\
/**
 * Population
 * Class
 * 
 * A population of neural networks.
 * 
 * @author Joshua Savoie
 */
public class Population {
    
    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private final String m_manifestFolder;
    
    private int m_generation;
    private Network[] m_networks;

    //-----[CONSTRUCTORS]-----\\
    /**
     * Population
     * Retrieve Population Constructor
     * 
     * Pass in the parent folder of the generation-to-pull.
     * Then pass in the generation-to-pulls's number.
     * 
     * @param manifestFolder
     * @param generationToPull
     */
    public Population(String manifestFolder, int generationToPull) {

        m_manifestFolder = manifestFolder;
        m_generation = generationToPull;

        // Gets the list of folders from the generation.
        File generationFolder = new File(getGenerationFolder());
        File[] networkFolders = generationFolder.listFiles();

        // Pulls the neural network population.
        m_networks = new Network[networkFolders.length];
        for(int network = 0; network < m_networks.length; network++) {
            m_networks[network] = new Network(networkFolders[network].getAbsolutePath());
        }

    }

    /**
     * Population
     * New Population Constructor
     * 
     * Creates a new neural network population of the given population size.
     * 
     * @param manifestFolder
     * @param populationSize
     * @param layerSizes
     * @param numInputs
     * @param function
     */
    public Population(String manifestFolder, int populationSize, int[] layerSizes, int numInputs, Layer.Functions function) {

        m_manifestFolder = manifestFolder;
        m_generation = 0;

        // Creates the neural network population.
        m_networks = new Network[populationSize];
        for(int network = 0; network < m_networks.length; network++) {
            m_networks[network] = new Network(
                getNetworkFolder(network), 
                numInputs, 
                layerSizes, 
                function
            );
        }

    }

    //-----[METHODS]-----\\

    /**
     * repopulate()
     * 
     * Repopulates the population pool of neural networks using the "graduated" neural networks as templates for the next generation.
     * 
     * @param graduationSize
     * @param costs
     */
    public void repopulate(int graduationSize, double[] costs) {
        
        assert graduationSize < m_networks.length: " The graduation size exceeds the size of the neural network population.";
        assert costs.length == m_networks.length: " The number of costs does not match the number of neural networks.";

        // Increments the generation.
        m_generation++;

        // Neural network templates.
        Network[] graduationClass = new Network[graduationSize];
        // The costs of each graduated neural network.
        // 1 neural network : 1 cost
        double[] graduationCosts = new double[graduationSize];
        // The new neural network population pool.
        Network[] newNetworks = new Network[m_networks.length];

        // These lists will be utilized as an ArrayList of Pair<T> abstractly.
        ArrayList<Network> populationAsList = new ArrayList<Network>(Arrays.asList(m_networks));
        ArrayList<Double> costsAsList = new ArrayList<Double>();
        // Due to costs being of the primitive datatype, values had to be entered in manually as opposed through Arrays.asList().
        for(double c : costs) costsAsList.add(c);

        // Fetches the neural networks that graduated.
        // The neural networks with the lowests costs are selected.
        for(int g = 0; g < graduationSize; g++) {

            // Searches for current lowest cost.
            // Stores the index of that neural network.
            int i = 0;
            double lowestCost = costsAsList.get(0);
            for(int p = 1; p < populationAsList.size(); p++) {

                if(costsAsList.get(p) < lowestCost) {
                    i = p;
                    lowestCost = costsAsList.get(p);
                }

            }

            // Stores the graduated neural network and its respective cost.
            graduationClass[g] = populationAsList.get(i);
            graduationCosts[g] = costsAsList.get(i);

            // Pops the graduated neural network and its respective cost from the ArrayLists.
            populationAsList.remove(i);
            costsAsList.remove(i);

        }

        // Uses the graduated neural networks as templates to create the new neural network population pool.
        newNetworks[0] = graduationClass[0].copy();
        newNetworks[0].setManifestFolder(getNetworkFolder(0));
        for(int p = 1; p < newNetworks.length; p += graduationSize) {

            for(int g = 0; g < graduationClass.length; g++) {
            
                if(p + g < newNetworks.length) {

                    newNetworks[p + g] = graduationClass[g].copy();
                    newNetworks[p + g].learn(graduationCosts[g]);

                    newNetworks[p + g].setManifestFolder(getNetworkFolder(p + g));

                }

            }

        }

        // Sets the population pool to the new pool.
        m_networks = newNetworks;

    }

    /**
     * getGenerationFolder()
     * 
     * @return
     */
    public String getGenerationFolder() {
        // If you wish to disable the saving of neural networks to multiple generation folders, replace "m_generation" with "1".
        return m_manifestFolder + "\\Generation_" + String.format("%03d", m_generation);
    }

    /**
     * getNetworkFolder()
     * 
     * @param network
     * @return
     */
    public String getNetworkFolder(int network) {
        return getGenerationFolder() + "\\Network_" + String.format("%03d", network);
    }

    /**
     * save()
     * 
     * Saves all neural networks to local.
     */
    public void save() {

        for(int n = 0; n < m_networks.length; n++) {

            m_networks[n].setManifestFolder(getNetworkFolder(n));
            m_networks[n].save();

        }

    }

    /**
     * getSize()
     * 
     * Returns the size of the population.
     * 
     * @return
     */
    public int getSize() {
        return m_networks.length;
    }

    /**
     * getNetworks()
     * 
     * NOTE: 
     * These are not copies of the population and are mutable.
     * Changes done to the returned networks affects the population.
     * 
     * @return
     */
    public Network[] getNetworks() {
        return m_networks;
    }

    /**
     * getGeneration()
     * 
     * @return
     */
    public int getGeneration() {
        return m_generation;
    }
    
}
