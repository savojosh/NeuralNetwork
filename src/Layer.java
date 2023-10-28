//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Layer
 * Class
 * 
 * A layer of nodes in the neural network.
 * 
 * @author Joshua Savoie
 */
public class Layer {

    //-----[CONSTANTS]-----\\

    // Used for weight regularization and generation
    private static final double UPPER_THRESHOLD = 1.0;
    private static final double LOWER_THRESHOLD = -1.0;

    //-----[VARIABLES]-----\\

    // Text file to store data to
    private String m_manifestFile;

    // The number of nodes
    private final int m_layerSize;
    // The number of inputs
    private final int m_numInputs;

    // Biases for each node
    private double[] m_biases;
    // Weights for each node to each input
    private double[][] m_weights;

    // The change for each bias to descend the gradient slope
    private double[] m_biasesGradient;
    // The change for each weight to each input to descend the gradient slope
    private double[][] m_weightsGradient;

    // The most recent outputs of the layer without applying the activation function
    private double[] m_zVector;
    // The most recent outputs with the activation function applied
    private double[] m_outputVector;

    //-----[CONSTRUCTORS]-----\\

    /**
     * Layer
     * Constructor
     * 
     * Creates a neural network layer.
     * 
     * @param manifestFile - Text file to store to.
     * @param layerSize - Size of the layer.
     * @param numInputs - The number of inputs going into the layer.
     */
    public Layer(String manifestFile, int layerSize, int numInputs) {

        m_manifestFile = manifestFile;

        m_layerSize = layerSize;
        m_numInputs = numInputs;

        m_biases = new double[m_layerSize];
        m_weights = new double[m_layerSize][m_numInputs];

        m_biasesGradient = new double[m_layerSize];
        m_weightsGradient = new double[m_layerSize][m_numInputs];

        // Sets all of the initial values for the weights and biases
        for(int n = 0; n < m_layerSize; n++) {

            m_biases[n] = 0;
            m_biasesGradient[n] = 0;

            for(int w = 0; w < m_numInputs; w++) {
                // Generates a weight value between the upper and lower thresholds.
                m_weights[n][w] = Math.random() * (UPPER_THRESHOLD - LOWER_THRESHOLD) + LOWER_THRESHOLD;
                m_weightsGradient[n][w] = 0;
            }
        }
    }

    /**
     * Layer
     * Constructor
     * 
     * Constructor only meant for copying a layer.
     * 
     * @param manifestFile
     * @param layerSize
     * @param numInputs
     * @param biases
     * @param weights
     * @param biasesGradient
     * @param weightsGradient
     * @param zVector
     * @param outputVector
     */
    private Layer(
        String manifestFile, int layerSize, int numInputs, 
        double[] biases, double[][] weights, 
        double[] biasesGradient, double[][] weightsGradient, 
        double[] zVector, double[] outputVector
    ) {
        m_manifestFile = manifestFile;
        m_layerSize = layerSize;
        m_numInputs = numInputs;
        m_biases = biases;
        m_weights = weights;
        m_biasesGradient = biasesGradient;
        m_weightsGradient = weightsGradient;
        m_zVector = zVector;
        m_outputVector = outputVector;
    }

    //-----[METHODS]-----\\

    /**
     * calculate()
     * 
     * Given some inputs, calculate the outputs for each node in this layer.
     * 
     * @param inputs - The input values into the layer.
     * @return
     */
    public double[] calculate(double[] inputs) {

        assert m_numInputs == inputs.length: " the input size of the layer and the number of inputs do not match.";

        // Outputs with activation function
        double[] outputs = new double[m_layerSize];
        // Outputs without activation function
        m_zVector = new double[m_layerSize];

        // summation(w * i + b)
        for(int n = 0; n < m_layerSize; n++) {
            
            double out = m_biases[n];

            for(int i = 0; i < m_numInputs; i++) {
                out += (m_weights[n][i] * inputs[i]);
            }

            m_zVector[n] = out;

            // Applies the activation function
            outputs[n] = Functions.bipolarSigmoid(out);

        }

        // Stores a copy of the output vector with the activation function applied
        m_outputVector = outputs.clone();

        return outputs;

    }

    /**
     * updateGradient()
     * 
     * Updates the gradient slope according to this layer's (L) error and the previous layer's (L-1) activations.
     * 
     * @param errors - This layer's (L) error.
     * @param previousActivations - The previous layer's (L-1) activations.
     */
    public void updateGradient(double[] errors, double[] previousActivations) {

        assert errors.length == m_layerSize: " the number of errors does not match the number of nodes.";
        assert previousActivations.length == m_numInputs: " the number of activations provided does not match the number of inputs to this layer.";
        
        for(int n = 0; n < m_layerSize; n++) {

            // dC/db = error
            // Change of cost in terms of bias equals the node's error
            m_biasesGradient[n] += (errors[n]);

            // dC/dw = activation * error
            // Change of cost in terms of weight equals the previous layer's activation multiplied by the node's error
            for(int w = 0; w < m_numInputs; w++) {
                m_weightsGradient[n][w] += (previousActivations[w] * errors[n]);
            }
        }

    }

    /**
     * applyGradient()
     * 
     * Applies the gradient to the layer.
     * 
     * @param miniBatchSize - The size of the training mini-batch.
     * @param learnRate - The rate to descend the gradient slope.
     */
    public void applyGardient(int miniBatchSize, double learnRate) {

        for(int n = 0; n < m_layerSize; n++) {

            // Change
            double d = 0;

            // Averages the gradient against the size of the mini-batch size.
            // Equivalent to storing all of the gradients from each training data point but saves a lot of space.
            d = m_biasesGradient[n] / miniBatchSize * learnRate;

            // Bias regularization after exceeding a threshold
            if(
                (m_biases[n] > UPPER_THRESHOLD && d > 0) ||
                (m_biases[n] < LOWER_THRESHOLD && d < 0)
            ) {
                m_biases[n] -= (d * (1.0 / Math.abs(m_biases[n])));
            } else {
                m_biases[n] -= d;
            }

            m_biasesGradient[n] = 0;

            for(int w = 0; w < m_numInputs; w++) {

                d = m_weightsGradient[n][w] / miniBatchSize * learnRate;

                // Weight regularization after exceeding a threshold
                if(
                    (m_weights[n][w] > UPPER_THRESHOLD && d > 0) ||
                    (m_weights[n][w] < LOWER_THRESHOLD && d < 0)
                ) {
                    m_weights[n][w] -= (d * (1.0 / Math.abs(m_weights[n][w])));
                } else {
                    m_weights[n][w] -= d;
                }

                m_weightsGradient[n][w] = 0;

            }
        }
    }

    /**
     * getZVector()
     * 
     * Gets the outputs of the layer without the activation function applied.
     * 
     * @return
     */
    public double[] getZVector() {
        return m_zVector.clone();
    }

    /**
     * getOutputVector()
     * 
     * Gets the outputs of the layeer with the activation function applied.
     * 
     * @return
     */
    public double[] getOutputVector() {
        return m_outputVector.clone();
    }

    /**
     * getWeights()
     * 
     * Gets the weights of the layer.
     * 
     * @return
     */
    public double[][] getWeights() {
        return m_weights.clone();
    }

    /**
     * getLayerSize()
     * 
     * Gets the size of the layer.
     * 
     * @return
     */
    public int getLayerSize() {
        return m_layerSize;
    }

    /**
     * copy()
     * 
     * Returns a copy of the scoped layer.
     * 
     * @return
     */
    public Layer clone() {
        return new Layer(
            new String(m_manifestFile), m_layerSize, m_numInputs, 
            m_biases.clone(), m_weights.clone(), 
            m_biasesGradient.clone(), m_weightsGradient.clone(), 
            m_zVector.clone(), m_outputVector.clone()
        );
    }

}