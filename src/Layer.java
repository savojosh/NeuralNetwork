//-----[IMPORTS]-----\\

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

//-----[CLASS]-----\\
/**
 * Layer
 * Class
 * 
 * Serves as a group of nodes taking in a certain number of inputs and returning a certain number of outputs.
 * These groupings of nodes can be understood as the "curve" of a line where each node is a value on that line.
 * 
 * @author Joshua Savoie
 */
public class Layer {

    //-----[CONSTANTS]-----\\

    /**
     * OutputFunctions
     * Enumeration
     * 
     * Allows layers to be versatile in that they can have different output functions applied to their inputs.
     */
    public static enum Functions {

        //-----[ENUM CONSTANTS]-----\\

        BINARY_STEP("BINARY_STEP"),
        SIGMOID("SIGMOID");

        //-----[ENUM DEFINITION]-----\\

        // Each layer saves the saveString at the top of their save files to denote the function used.
        // Combined with an enum, the functions and save strings are standardized.
        public final String saveString;

        private Functions(String saveString) {
            this.saveString = saveString;
        }

    }

    // MIN AND MAX CHANGE FOR WEIGHTS AND BIASES
    private static final double MAX = 1;
    private static final double MIN = -1;

    // How precise biases and weights should be.
    // 7 zeros = 7 decimal places.
    private static final double PRECISION = 10000000;
    
    //-----[VARIABLES]-----\\

    // Save file.
    private String m_manifestFile;

    // Size of the layer.
    private int m_layerSize;
    // How many inputs the layer takes in.
    private int m_numInputs;
    // The function the layer uses to calculate the node vector.
    private Functions m_function;

    // The bias for each node.
    private double[] m_biases;
    // The weight for each input to every node.
    private double[][] m_weights;
    // The output values of each node when the layer's nodes were last calculated.
    private double[] m_vector;

    //-----[CONSTRUCTORS]-----\\

    /**
     * Layer
     * Copy Constructor
     * 
     * Used by copy()
     */
    private Layer() {}

    /**
     * Layer
     * Retrieve Layer Constructor
     * 
     * Retrieves an existing layer of a network from a given manifest file.
     * 
     * @param manifestFile
     */
    public Layer(String manifestFile) {

        m_manifestFile = manifestFile;
        
        try {

            // Get file lines.

            Scanner reader = new Scanner(new File(m_manifestFile));

            ArrayList<String> lines = new ArrayList<String>();

            while(reader.hasNextLine()) {
                lines.add(reader.nextLine().strip());
            }

            // Gets the layer size (each line represents a node according to the save() function).
            // 1 is subtracted due to the first line of the file being taken by the layer's activation function.
            m_layerSize = lines.size() - 1;
            // Gets the input size based on how many weights are present for the first node.
            m_numInputs = lines.get(1).split(":")[1].split(",").length;
            
            // Searches for the function saved in the first line of the save file.
            m_function = null;
            for(Functions function : Functions.values()) {
                if(lines.get(0).equals(function.saveString)) {
                    m_function = function;
                }
            }
            if(m_function == null) {
                throw new Exception("No activation function found.");
            }
            
            // Initializes the arrays to store layer-significant values.
            m_vector = new double[m_layerSize];
            m_biases = new double[m_layerSize];
            m_weights = new double[m_layerSize][m_numInputs];

            // Loops through the lines and gets the vector, bias, and weight values for each node.
            for(int node = 0; node < m_layerSize; node++) {

                String line = lines.get(node + 1);

                String first = line.split(":")[0];
                String stringV = first.split(",")[0];
                String stringB = first.split(",")[1];

                String second = line.split(":")[1];
                String[] stringWs = second.split(",");

                m_vector[node] = Double.parseDouble(stringV);
                m_biases[node] = Double.parseDouble(stringB);

                for(int weight = 0; weight < m_numInputs; weight++) {

                    m_weights[node][weight] = Double.parseDouble(stringWs[weight]);

                }

            }

        } catch(Exception e) {

            System.out.println(e.getMessage());

            System.exit(0);

        }

    }

    /**
     * Layer
     * New Layer Constructor
     * 
     * Creates a neural network layer (thus creating curve) to add complexity to a network's functionality and resulting output(s).
     * 
     * @param manifestFile
     * @param layerSize
     * @param numInputs
     * @param function
     */
    public Layer(String manifestFile, int layerSize, int numInputs, Functions function) {

        m_manifestFile = manifestFile;

        m_layerSize = layerSize;
        m_numInputs = numInputs;
        m_function = function;

        m_biases = new double[m_layerSize];
        m_weights = new double[m_layerSize][m_numInputs];

        // Sets all biases to 0.
        for(int bias = 0; bias < m_layerSize; bias++) {
            m_biases[bias] = 0;
        }

        // Randomly generates a weight for each input of each node ranging from m_MAX to m_MIN.
        for(int node = 0; node < m_layerSize; node++) {

            for(int weight = 0; weight < m_numInputs; weight++) {

                m_weights[node][weight] = Math.round((Math.random() * (MAX - MIN) + MIN) * PRECISION) / PRECISION;

            }

        }
    }

    //-----[METHODS]-----\\

    /**
     * learn()
     * 
     * Changes all of the weights and biases synchronously with 
     * the actual and desired scores acting as a multiplier to the changes.
     * 
     * @param cost = Acts as a multiplier to changes.
     */
    public void learn(double cost) {

        // Updates the biases.
        for(int bias = 0; bias < m_layerSize; bias++) {

            double change = Math.random() * (MAX - MIN) + MIN;

            if((m_biases[bias] > MAX && change > 0) || (m_biases[bias] < MIN && change < 0)) {
                // bias regularization
                // due to the nature of the sigmoid function, if a bias becomes too large, either positive or negative, 
                // the f will always return 0 or 1 with no in-between.
                change = change * cost * 1.0 / Math.sqrt(Math.pow(m_biases[bias], 2));
            } else {
                change = change * cost;
            }
            
            m_biases[bias] = Math.round((m_biases[bias] + change) * PRECISION) / PRECISION;

        }

        // Updates the weights.
        for(int node = 0; node < m_layerSize; node++) {

            for(int weight = 0; weight < m_numInputs; weight++) {

                double change = Math.random() * (MAX - MIN) + MIN;

                if((m_weights[node][weight] > MAX && change > 0) || (m_weights[node][weight] < MIN && change < 0)) {
                    // weight regularization
                    // due to the nature of the sigmoid function, if a weight becomes too large, either positive or negative, 
                    // the weight * input will always result in 0 or 1 from the sigmoid function.
                    change = change * cost * 1.0 / Math.sqrt(Math.pow(m_weights[node][weight], 2));
                } else {
                    change = change * cost;
                }
                
                m_weights[node][weight] = Math.round((m_weights[node][weight] + change) * PRECISION) / PRECISION;

            }

        }

    }

    /**
     * calculate()
     * 
     * Calculates the outputs for a neural network layer based on given inputs and a given activation function.
     * 
     * Inputs should doubles ranging from -1.000 to +1.000 in value.
     * 
     * @param inputs
     * @return
     */
    public double[] calculate(double[] inputs) {

        assert m_numInputs == inputs.length: " the input size of the layer and the number of inputs do not match.";

        // The output vector.
        double[] outputs = new double[m_layerSize];

        // Loops through the nodes and calculates each nodes output.
        for(int node = 0; node < m_layerSize; node++) {

            double output = m_biases[node];

            // output = summation(w * i) + bias
            for(int input = 0; input < m_numInputs; input++) {

                output += m_weights[node][input] * inputs[input];

            }

            // Applies an activation function to the output.
            switch(m_function) {

                case BINARY_STEP:
                    // 1 if the output >= 0.
                    // 0 if the output < 0.
                    outputs[node] = (output >= 0 ? 1 : 0);
                    break;

                case SIGMOID:
                    // y = 1 / (1 + e^(-x))
                    outputs[node] = 1 / (1 + Math.pow(Math.E, -output));
                    break;

            }

        }

        // Stores the output vector for history tracking in save files.
        m_vector = outputs.clone();
        return outputs;

    }

    /**
     * save()
     * 
     * Saves a layer to file.
     */
    public void save() {

        try {

            // Creates the manifest file if it doesn't already exist.
            File file = new File(m_manifestFile);
            file.createNewFile();

            // false = Overwrites any existing content of the file.
            FileWriter writer = new FileWriter(file, false);

            String output = new String("");
            String[] lines = getEntries();

            // Combines all of the entires into one string.
            for(int line = 0; line < lines.length; line++) {

                if(line == lines.length - 1) {

                    output += lines[line];

                } else {

                    output += lines[line] + "\n";

                }
            }

            // Writes the string to the file.
            writer.write(output);

            writer.close();

        } catch(Exception e) {

            System.out.println(e.getMessage());

            System.exit(0);

        }


    }

    /**
     * getEntries()
     * 
     * Returns a list of strings in "layer save" format.
     * 
     * @return
     */
    public String[] getEntries() {

        String[] entries = new String[m_layerSize + 1];

        // Stores the activation function.
        entries[0] = m_function.saveString;

        // Loops through the nodes and saves their value, bias, and weights.
        for(int node = 0; node < m_layerSize; node++) {

            entries[node + 1] = String.format("%.7f", m_vector[node]) + "," + String.format("%.7f", m_biases[node]) + ":";

            for(int weight = 0; weight < m_numInputs; weight++) {

                // If the last weight in the layer:
                if(weight == m_numInputs - 1) {
                    entries[node + 1] += String.format("%.7f", m_weights[node][weight]);

                // Else (not the last):
                } else {
                    entries[node + 1] += String.format("%.7f", m_weights[node][weight]) + ",";
                }

            }

        }

        return entries;

    }

    /**
     * setManifestFile()
     * 
     * @param manifestFile
     */
    public void setManifestFile(String manifestFile) {
        m_manifestFile = manifestFile;
    }

    /**
     * getInputSize()
     * 
     * @return
     */
    public int getInputSize() {
        return m_numInputs;
    }

    /**
     * getLayerSize()
     * 
     * @return
     */
    public int getLayerSize() {
        return m_layerSize;
    }

    /**
     * getFunction()
     * 
     * @return
     */
    public Functions getFunction() {
        return m_function;
    }

    //-----[COPY]-----\\

    /**
     * copy()
     * 
     * Returns a new instance of class Layer with all the same member variable values.
     * 
     * @return
     */
    public Layer copy() {

        Layer c = new Layer();

        c.copyManifest(m_manifestFile);
        c.copyLayerSize(m_layerSize);
        c.copyNumInputs(m_numInputs);
        c.copyFunction(m_function);
        c.copyBiases(m_biases);
        c.copyWeights(m_weights);
        c.copyVector(m_vector);

        return c;
    }

    /**
     * copyManifest()
     * 
     * @param toCopy
     */
    private void copyManifest(String toCopy) {
        m_manifestFile = new String(toCopy);
    }

    /**
     * copyLayerSize()
     * 
     * @param toCopy
     */
    private void copyLayerSize(int toCopy) {
        m_layerSize = toCopy;
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
     * copyFunction()
     * 
     * @param toCopy
     */
    private void copyFunction(Functions toCopy) {
        m_function = toCopy;
    }

    /**
     * copyBiases()
     * 
     * @param toCopy
     */
    private void copyBiases(double[] toCopy) {
        m_biases = new double[toCopy.length];
        for(int i = 0; i < toCopy.length; i++) {
            m_biases[i] = toCopy[i];
        }
    }

    /**
     * copyWeights()
     * 
     * @param toCopy
     */
    private void copyWeights(double[][] toCopy) {
        m_weights = new double[toCopy.length][toCopy[0].length];
        for(int x = 0; x < toCopy.length; x++) {
            for(int y = 0; y < toCopy[0].length; y++) {
                m_weights[x][y] = toCopy[x][y];
            }
        }
    }

    /**
     * copyVector()
     * 
     * @param toCopy
     */
    private void copyVector(double[] toCopy) {
        m_vector = new double[toCopy.length];
        for(int i = 0; i < toCopy.length; i++) {
            m_vector[i] = toCopy[i];
        }
    }

}
