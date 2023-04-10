//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * DataPoint
 * Class
 * 
 * Stores the relevant data to a specific case.
 * 
 * @author Joshua Savoie
 */
public class DataPoint {

    //-----[VARIABLES]-----\\
    
    public final int label;
    public final double[] values;
    public final double[] y;

    //-----[CONSTRUCTOR]-----\\
    /**
     * DataPoint
     * Constructor
     * 
     * Creates a data point.
     * 
     * @param label - The expected decision. Often the largest positive value in double[] y. 0 index'd.
     * @param values - The input values for the neural network.
     * @param y - The expected outputs from the neural network.
     */
    public DataPoint(int label, double[] values, double[] y) {

        // Get the max magnitude of a value.
        double max = 1;
        for(double v : values) {
            if(Math.abs(v) > max) {
                max = Math.abs(v);
            }
        }

        // Turn all values to decimals.
        for(int v = 0; v < values.length; v++) {
            values[v] = values[v] / max;
        }

        this.label = label;
        this.values = values;
        this.y = y;

    }

}
