//-----[IMPORTS]-----\\

import java.io.File;

//-----[CLASS]-----\\
/**
 * Network
 * Class
 * 
 * Serves as a group of layers that work together to evaluate given inputs and calculate the best possible
 * decision from those given inputs.
 * 
 * @author Joshua Savoie
 */
public class Network {
    
    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    // Folder to save the layers too.
    private String m_manifestFolder;

    // Number of layers.
    private int m_networkSize;
    // Number of inputs (otherwise referred to the input layer or 0th layer).
    private int m_numInputs;

    // Each of the hidden layers and the output layer.
    private Layer[] m_layers;

    // Activation function.
    private Layer.Functions m_function;

    //-----[CONSTRUCTORS]-----\\

    /**
     * Network
     * Copy Constructor
     * 
     * Used by copy()
     */
    private Network() {}

    /**
     * Network
     * Retrieve Network Constructor
     * 
     * This constructor is used to retrieve an existing network from a manifest folder.
     * 
     * NOTE: The last layer acts as the output layer.
     * NOTE: Using layers that don't follow the naming conventions in the New Network Constructor may result in errors.
     * 
     * @param manifestFolder = The location where the network is saved at. Must be an absolute path.
     */
    public Network(String manifestFolder) {

        // Location for all network layer files to be stored.
        m_manifestFolder = manifestFolder;

        // Fetching the manifest folder and its contents.
        File folder = new File(m_manifestFolder);
        File[] files = folder.listFiles();

        // Declaring the layers array
        m_networkSize = files.length;
        m_layers = new Layer[m_networkSize];

        // Looping through the files and literally creating the layers.
        for(int layer = 0; layer < m_networkSize; layer++) {
            m_layers[layer] = new Layer(files[layer].getAbsolutePath());
        }

        // Gets the number of inputs taken in by the network.
        m_numInputs = m_layers[0].getInputSize();

        m_function = m_layers[0].getFunction();

    }

    /**
     * Network
     * New Network Constructor
     * 
     * Creates a new neural network with a given number of inputs and layer sizes.
     * 
     * NOTE: The last layer acts as the output layer.
     * 
     * @param manifestFolder = The location to save the network to when Network.save() is called. Must be an absolute path.
     * @param numInputs
     * @param layerSizes
     * @param functions = The function used to calculate outputs.
     */
    public Network(String manifestFolder, int numInputs, int[] layerSizes, Layer.Functions function) {

        // Location for all network layer files to be stored.
        m_manifestFolder = manifestFolder;

        m_networkSize = layerSizes.length;
        m_numInputs = numInputs;

        m_layers = new Layer[m_networkSize];

        m_function = function;

        // Creating all of the layers
        for(int layer = 0; layer < m_networkSize; layer++) {

            // If it is the first layer, use the number of inputs that was passed in.
            if(layer == 0) {

                m_layers[layer] = new Layer(
                    m_manifestFolder + "\\Layer_" + String.format("%03d", layer + 1),
                    layerSizes[layer],
                    m_numInputs,
                    m_function
                );

            // Else (not the first) pull the number of inputs for the layer from the previous layer.
            } else {

                m_layers[layer] = new Layer(
                    m_manifestFolder + "\\Layer_" + String.format("%03d", layer + 1),
                    layerSizes[layer],
                    layerSizes[layer - 1],
                    m_function
                );
                
            }
        }

    }

    //-----[METHODS]-----\\

    /**
     * cost()
     * 
     * Calculates the cost of a neural network's outputs compared to the expected outputs.
     * 
     * @param actualOutputs
     * @param expectedOutputs
     * @return
     */
    public double cost(double[] actualOutputs, double[] expectedOutputs) {

        assert actualOutputs.length == expectedOutputs.length: " output arrays do not match in length.";

        double cost = 0;

        for(int out = 0; out < actualOutputs.length; out++) {

            cost += Math.pow(expectedOutputs[out] - actualOutputs[out], 2);

        }

        return cost;

    }

    public double[] error(double[] actualOutputs, double[] expectedOutputs) {

        assert actualOutputs.length == expectedOutputs.length: " output arrays do not match in length.";

        double[] errorVector = new double[actualOutputs.length];

        for(int i = 0; i < errorVector.length; i++) {

            switch(m_function) {

                case BINARY_STEP:

                    break;
                
                case SIGMOID:
                    errorVector[i] = 2 * (actualOutputs[i] - expectedOutputs[i]);
                    errorVector[i] *= (1 / (1 + Math.pow(Math.E, -actualOutputs[i])));
                    errorVector[i] *= (1 - 1 / (1 + Math.pow(Math.E, -actualOutputs[i])));
                    break;

            }

        }

        return errorVector;

    }

    /**
     * learn()
     * 
     * Updates all weights and biases synchronously in all layers of the network.
     * 
     * @param cost = Acts as a multiplier to changes.
     */
    public void learn(double cost) {

        // Loops through all of the layers and updates them.
        for(int layer = 0; layer < m_networkSize; layer++) {

            m_layers[layer].learn(cost);

        }

    }

    /**
     * calculate()
     * 
     * Takes in inputs, runs those inputs through the network, and returns a list of outputs.
     * 
     * These outputs can be used to make a decision.
     * 
     * @param inputs
     * @return
     * @throws Exception
     */
    public double[] calculate(double[] inputs) {

        assert m_numInputs == inputs.length: " The set number of inputs does not match the actual number of inputs.";

        // The last layer is the output layer.
        double[] outputs = new double[m_layers[m_layers.length - 1].getLayerSize()];

        // Loops through the layers and calculates the outputs of each.
        for(int layer = 0; layer < m_networkSize; layer++) {

            // If the first layer, calculate using the passed in inputs.
            if(layer == 0) {

                outputs = m_layers[layer].calculate(inputs);

            // Else (not the first) calculate using the outputs from the previous layer.
            } else {

                outputs = m_layers[layer].calculate(outputs);

            }

        }

        return outputs;

    }

    /**
     * save()
     * 
     * Stores the network and all of its layers into the manifest folder.
     * 
     * Creates the manifest folder and all of its parent directories should it not exist.
     */
    public void save() {

        try {

            // Creates all directories specified in the manifest folder.
            File folder = new File(m_manifestFolder);
            folder.mkdirs();

            for(int layer = 0; layer < m_networkSize; layer++) {

                m_layers[layer].save();

            }

        } catch(Exception e) {

            System.out.println(e.getMessage());

            System.exit(0);

        }

    }

    /**
     * setManifestFolder
     * 
     * Sets the manifest folder to a location.
     * 
     * @param manifestFolder = Absolute path to the folder.
     */
    public void setManifestFolder(String manifestFolder) {

        m_manifestFolder = manifestFolder;

        for(int layer = 0; layer < m_networkSize; layer++) {

            m_layers[layer].setManifestFile(m_manifestFolder + "\\Layer_" + String.format("%03d", layer + 1));

        }

    }

    //-----[COPY]-----\\

    /**
     * copy()
     * 
     * Returns a new instance of class Network with all the same member variable values.
     * 
     * @return
     */
    public Network copy() {

        Network c = new Network();

        c.copyManifest(m_manifestFolder);
        c.copyNetworkSize(m_networkSize);
        c.copyNumInputs(m_numInputs);
        c.copyLayers(m_layers);

        return c;

    }

    /**
     * copyManifest()
     * 
     * @param toCopy
     */
    private void copyManifest(String toCopy) {
        m_manifestFolder = new String(toCopy);
    }

    /**
     * copyNetworkSize()
     * 
     * @param toCopy
     */
    private void copyNetworkSize(int toCopy) {
        m_networkSize = toCopy;
    }

    /**
     * copyNumInputs()
     * 
     * @param toCopy
     */
    private void copyNumInputs(int toCopy) {
        m_numInputs = toCopy;
    }

    /**
     * copyLayers()
     * 
     * @param toCopy
     */
    private void copyLayers(Layer[] toCopy) {
        m_layers = new Layer[toCopy.length];
        for(int i = 0; i < toCopy.length; i++) {
            m_layers[i] = toCopy[i].copy();
        }
    }

}
