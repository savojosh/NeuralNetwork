//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Network
 * Class
 * 
 * A perceptron neural network.
 */
public class Network {

    //-----[VARIABLES]-----\\
    
    // Folder location to save all layer files to
    private String m_manifestFolder;

    // The number of layers
    private final int m_networkSize;
    // The number of incoming inputs into the neural network as a whole
    private final int m_numInputs;

    private Layer[] m_layers;

    //-----[CONSTRUCTORS]-----\\

    /**
     * Network
     * Constructor
     * 
     * Creates a perception neural network.
     * 
     * @param manifestFolder - Folder location to save all layer files to.
     * @param numInputs - The number of inputs into the neural network.
     * @param layerSizes - The layers of each layer in the network. Does not include the input layer.
     */
    public Network(String manifestFolder, int numInputs, int[] layerSizes) {

        m_manifestFolder = manifestFolder;

        m_networkSize = layerSizes.length;
        m_numInputs = numInputs;

        // Creating all of the layers
        m_layers = new Layer[m_networkSize];

        m_layers[0] = new Layer(
            m_manifestFolder + "\\Layer_" + String.format("%03d", 1),
            layerSizes[0],
            m_numInputs
        );
        for(int l = 1; l < m_networkSize; l++) {

            m_layers[l] = new Layer(
                m_manifestFolder + "\\Layer_" + String.format("%03d", l + 1),
                layerSizes[l],
                layerSizes[l - 1]
            );

        }

    }

    /**
     * Network()
     * Constructor
     * 
     * Constructor only meant for copying a network.
     * 
     * @param manifestFolder
     * @param networkSize
     * @param numInputs
     * @param layers
     */
    private Network(String manifestFolder, int networkSize, int numInputs, Layer[] layers) {
        m_manifestFolder = manifestFolder;
        m_networkSize = networkSize;
        m_numInputs = numInputs;
        m_layers = layers;
    }

    //-----[METHODS]-----\\

    /**
     * calculate()
     * 
     * Calculates an output based on the given data point.
     * 
     * @param dp - Data point.
     * @return
     */
    public double[] calculate(DataPoint dp) {

        assert dp.values.length == m_numInputs: " the number of inputs does not equal the number of inputs accepted by the network.";
        assert dp.y.length == m_layers[m_layers.length - 1].getLayerSize(): " the number of outputs does not match the number of outputs returned by the network.";

        double[] outputs = m_layers[0].calculate(dp.values);
        for(int l = 1; l < m_networkSize; l++) {
            outputs = m_layers[l].calculate(outputs);
        }

        return outputs;

    }

    /**
     * learn()
     * 
     * Given some batch of data and learn rate, create a gradient slope
     * and update the biases and weights of each layer accordingly.
     * 
     * @param data - Data to create a gradient slope from.
     * @param learnRate - The rate to descend the gradient slope.
     * @return
     */
    public double[][] learn(DataPoint[] data, double learnRate) {

        // The outputs for each data point
        double[][] outputs = new double[data.length][data[0].y.length];
        // Used to calculate the average cost over all given data points
        double cost = 0;

        // Loop through the data points
        for(int dp = 0; dp < data.length; dp++) {

            outputs[dp] = calculate(data[dp]);

            // Handles the output layer gradient
            double[] errors = new double[m_layers[m_layers.length - 1].getLayerSize()];
            for(int o = 0; o < data[dp].y.length; o++) {

                double a = outputs[dp][o];
                double y = data[dp].y[o];
                double z = m_layers[m_layers.length - 1].getZVector()[o];

                // dC/da * derivative of the activation function
                errors[o] = (2 * (a - y)) * (Functions.dSigmoid(z));

                cost += Math.pow(a - y, 2);

            }
            
            // Checks if the number of layers is greater than 1
            if(m_layers.length > 1) {

                // Updates the output layer with its errors and the activations from the prior layer's output.
                m_layers[m_layers.length - 1].updateGradient(errors, m_layers[m_layers.length - 2].getOutputVector());

                // Handles all intermediate/hidden layers' gradients
                for(int l = m_layers.length - 2; l >= 0; l--) {

                    double[] previousErrors = errors.clone();
                    double[][] outWeights = m_layers[l + 1].getWeights();
                    errors = new double[m_layers[l].getLayerSize()];

                    /*
                     * The transpose of layer[L+1]'s weights multiplied against layer[L+1]'s errors
                     * That product is then applied to a hadamard multiplication with layer[L]'s output 
                     * vector (with no activation function applied)
                     * The resulting hadamard product vector is an array of errors for layer[L]
                     * This can be thought of as "moving the error backward through the network" - Nielsen's
                     * work "Neural Networks and Deep Learning"
                     */
                    // Current node of layer[L]
                    for(int cn = 0; cn < errors.length; cn++) {

                        errors[cn] = 0;

                        // "Next" node of layer[L+1]
                        for(int nn = 0; nn < outWeights.length; nn++) {

                            double w = outWeights[nn][cn];
                            double e = previousErrors[nn];
                            double z = m_layers[l].getZVector()[cn];

                            /*
                             * Essentially sums all of the weights of layer[L+1] that comes from all of layer[L]'s cn node.
                             * layer[L+1]'s node[nn]'s weight to layer[L]'s cn node multiplied by layer[L+1]'s node[nn]'s error
                             * multiplied by the derivative of the activation function calculated with layer[L]'s node[cn]'s
                             * output without the activation function applied.
                             */
                            errors[cn] += w * e * Functions.dSigmoid(z);

                        }

                        errors[cn] = errors[cn] / outWeights.length;
                        
                    }

                    // Special case for l=0 as it receives input from the data point itself
                    if(l > 0) {
                        m_layers[l].updateGradient(errors, m_layers[l - 1].getOutputVector());
                    } else {
                        m_layers[l].updateGradient(errors, data[dp].values);
                    }

                }
                
            // If the number of layers was 1...
            } else {
                m_layers[m_layers.length - 1].updateGradient(errors, data[dp].values);
            }

        }

        // Loops through all of the layers and applies their gradients
        for(int l = m_layers.length - 1; l >= 0; l--) {

            m_layers[l].applyGardient(data.length, learnRate);

        }

        return outputs;

    }

    /**
     * clone()
     * 
     * Creates a copy of this neural network.
     */
    public Network clone() {

        Layer[] newLayers = new Layer[m_layers.length];
        for(int l = 0; l < newLayers.length; l++) {
            newLayers[l] = m_layers[l].clone();
        }

        return new Network(new String(m_manifestFolder), m_networkSize, m_numInputs, m_layers);

    }

}
