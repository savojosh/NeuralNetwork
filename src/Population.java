//-----[IMPORTS]-----\\

import java.util.Arrays;
import java.util.ArrayList;

//-----[CLASS]-----\\
/**
 * Population
 * Class
 * 
 * A population of neural networks.
 */
public class Population {
    
    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private final String m_manifestFolder;
    
    private int m_generation = 0;
    private Network[] m_networks;

    //-----[CONSTRUCTORS]-----\\

    public Population(String manifestFolder, int populationSize, int[] layerSizes, int numInputs, Layer.Functions function) {

        m_manifestFolder = manifestFolder;

        m_networks = new Network[populationSize];
        for(int network = 0; network < m_networks.length; network++) {
            m_networks[network] = new Network(
                m_manifestFolder + "\\Layer_" + String.format("%03d", network + 1), 
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
        newNetworks[0] = graduationClass[0];
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

        // Increments the generation.
        m_generation++;

    }

    /**
     * getGenerationFolder()
     * 
     * @return
     */
    public String getGenerationFolder() {
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

    // public <A, B> ArrayList<Object[]> unionToPairs(A[] a, B[] b) {

    //     assert a.length == b.length: " The length of 'a' does not match the length of 'b'.";

    //     ArrayList<Object[]> l = new ArrayList<Object[]>(a.length);

    //     for(int i = 0; i < l.size(); i++) {
            
    //         Object[] o = new Object[2];
    //         o[0] = a[i];
    //         o[1] = b[i];
    //         l.add(o);

    //     }

    //     return l;

    // }

}
