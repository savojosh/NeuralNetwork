public class Functions {
    
    public static double sigmoid(double x) {
        return (double)(1 / (1 + (double)Math.pow(Math.E, -x)));
    }
    public static double dSigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

}
