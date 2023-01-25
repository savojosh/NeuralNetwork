public class Network {
    
    private final int m_networkSize;
    private final int m_numInputs;

    private String m_manifestFolder;

    private Layer[] m_layers;

    public Network(String manifestFolder, int numInputs, int[] layerSizes) {

        m_manifestFolder = manifestFolder;

        m_networkSize = layerSizes.length;
        m_numInputs = numInputs;

        m_layers = new Layer[m_networkSize];

        m_layers[0] = new Layer(
            m_manifestFolder + "\\Layer_" + String.format("%03d", 1),
            layerSizes[0],
            m_numInputs
        );
        for(int l = 1; l < m_networkSize; l++) {

            m_layers[l] = new Layer(
                m_manifestFolder + "\\Layer_" + String.format("%03d", l),
                layerSizes[l],
                layerSizes[l - 1]
            );

        }

    }

    public double[] calculate(DataPoint dp) {

        assert dp.values.length == m_numInputs: " the number of inputs does not equal the number of inputs accepted by the network.";
        assert dp.y.length == m_layers[m_layers.length - 1].getLayerSize(): " the number of outputs does not match the number of outputs returned by the network.";

        double[] outputs = m_layers[0].calculate(dp.values);
        for(int l = 1; l < m_networkSize; l++) {
            outputs = m_layers[l].calculate(outputs);
        }

        return outputs;

    }

    public double[][] learn(DataPoint[] data, double learnRate) {

        double[][] outputs = new double[data.length][data[0].y.length];
        double[] cost = new double[data.length];

        for(int dp = 0; dp < data.length; dp++) {

            outputs[dp] = calculate(data[dp]);

            // handles the output layer gradient
            double[] errors = new double[m_layers[m_layers.length - 1].getLayerSize()];
            cost[dp] = 0;
            for(int o = 0; o < data[dp].y.length; o++) {
                double a = outputs[dp][o];
                double y = data[dp].y[o];
                double z = m_layers[m_layers.length - 1].getZVector()[o];
                errors[o] = 2 * (a - y) * Functions.dSigmoid(z);
                cost[dp] += Math.pow(a - y, 2);
            }
            cost[dp] = cost[dp] / data[dp].y.length;

            if(m_layers.length > 1) {
                m_layers[m_layers.length - 1].updateGradient(errors, m_layers[m_layers.length - 2].getOutputVector());

                // handles all intermediate layers' gradients
                for(int l = m_layers.length - 2; l >= 0; l--) {

                    double[] previousErrors = errors.clone();
                    double[][] outWeights = m_layers[l + 1].getWeights();
                    errors = new double[m_layers[l].getLayerSize()];

                    for(int cn = 0; cn < errors.length; cn++) {

                        double error = 0;

                        for(int nn = 0; nn < outWeights.length; nn++) {

                            double w = outWeights[nn][cn];
                            double e = previousErrors[nn];
                            double z = m_layers[l].getZVector()[cn];
                            error += w * e * Functions.dSigmoid(z);

                        }

                        errors[cn] = error / outWeights.length;
                        
                    }

                    if(l > 0) {
                        m_layers[l].updateGradient(errors, m_layers[l - 1].getOutputVector());
                    } else {
                        m_layers[l].updateGradient(errors, data[dp].values);
                    }

                }
                
            } else {
                m_layers[m_layers.length - 1].updateGradient(errors, data[dp].values);
            }

        }

        for(int l = m_layers.length - 1; l >= 0; l--) {

            m_layers[l].applyGardient(learnRate);

        }

        double avgCost = 0;
        for(double c : cost) {
            avgCost += c;
        }
        avgCost = avgCost / cost.length;
        System.out.println(avgCost);

        return outputs;

    }

}