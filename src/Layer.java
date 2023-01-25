import java.util.ArrayList;

public class Layer {

    private static final double UPPER_THRESHOLD = 1.0;
    private static final double LOWER_THRESHOLD = -1.0;
    // private static final double PRECISION = 1_000_000_000;

    private String m_manifestFile;

    private final int m_layerSize;
    private final int m_numInputs;

    private double[] m_biases;
    private double[][] m_weights;

    private double[] m_zVector;
    private double[] m_outputVector;

    private ArrayList<Double>[] m_biasesGradient;
    private ArrayList<Double>[][] m_weightsGradient;

    public Layer(String manifestFile, int layerSize, int numInputs) {

        m_manifestFile = manifestFile;

        m_layerSize = layerSize;
        m_numInputs = numInputs;

        m_biases = new double[m_layerSize];
        m_weights = new double[m_layerSize][m_numInputs];

        m_biasesGradient = new ArrayList[m_layerSize];
        m_weightsGradient = new ArrayList[m_layerSize][m_numInputs];

        for(int n = 0; n < m_layerSize; n++) {

            m_biases[n] = 0;
            m_biasesGradient[n] = new ArrayList<Double>();

            for(int w = 0; w < m_numInputs; w++) {
                m_weights[n][w] = Math.random() * (UPPER_THRESHOLD - LOWER_THRESHOLD) + LOWER_THRESHOLD;
                m_weightsGradient[n][w] = new ArrayList<Double>();
            }
        }

    }

    public double[] calculate(double[] inputs) {

        assert m_numInputs == inputs.length: " the input size of the layer and the number of inputs do not match.";

        double[] outputs = new double[m_layerSize];
        m_zVector = new double[m_layerSize];

        for(int n = 0; n < m_layerSize; n++) {
            
            double out = m_biases[n];

            for(int i = 0; i < m_numInputs; i++) {
                out += (m_weights[n][i] * inputs[i]);
            }

            m_zVector[n] = out;

            outputs[n] = Functions.sigmoid(out);

        }

        m_outputVector = outputs.clone();

        return outputs;

    }

    public void updateGradient(double[] errors, double[] previousActivations) {

        assert previousActivations.length == m_numInputs: " the number of activations provided does not match the number of inputs to this layer.";
        
        for(int n = 0; n < m_layerSize; n++) {

            m_biasesGradient[n].add(errors[n]);

            for(int w = 0; w < m_numInputs; w++) {
                m_weightsGradient[n][w].add(previousActivations[w] * errors[n]);
            }
        }

    }

    public void applyGardient(double learnRate) {

        for(int n = 0; n < m_layerSize; n++) {

            double d = 0;
            for(int g = 0; g < m_biasesGradient[n].size(); g++) {
                d += m_biasesGradient[n].get(g);
            }
            d = d / m_biasesGradient[n].size() * learnRate;

            if(
                (m_biases[n] > UPPER_THRESHOLD && d > 0) ||
                (m_biases[n] < LOWER_THRESHOLD && d < 0)
            ) {
                m_biases[n] -= (d * (1.0 / Math.abs(m_biases[n])));
            } else {
                m_biases[n] -= d;
            }

            m_biasesGradient[n].clear();

            for(int w = 0; w < m_numInputs; w++) {

                for(int g = 0; g < m_weightsGradient[n][w].size(); g++) {
                    d += m_weightsGradient[n][w].get(g);
                }
                d = d / m_weightsGradient[n][w].size() * learnRate;

                if(
                    (m_weights[n][w] > UPPER_THRESHOLD && d > 0) ||
                    (m_weights[n][w] < LOWER_THRESHOLD && d < 0)
                ) {
                    m_weights[n][w] -= (d * (1.0 / Math.abs(m_weights[n][w])));
                } else {
                    m_weights[n][w] -= d;
                }

                m_weightsGradient[n][w].clear();

            }
        }
    }

    public double[] getZVector() {
        return m_zVector.clone();
    }

    public double[] getOutputVector() {
        return m_outputVector.clone();
    }

    public double[][] getWeights() {
        return m_weights.clone();
    }

    public int getLayerSize() {
        return m_layerSize;
    }

}