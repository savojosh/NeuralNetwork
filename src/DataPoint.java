public class DataPoint {
    
    public final int label;
    public final double[] values;
    public final double[] y;

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
