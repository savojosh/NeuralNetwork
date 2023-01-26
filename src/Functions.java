//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Functions
 * Class
 * 
 * Provides static methods for all functions.
 * 
 * @author Joshua Savoie
 */
public class Functions {
    
    //-----[SIGMOID]-----\\

    /**
     * sigmoid()
     * 
     * Regular sigmoid.
     * 
     * @param x
     * @return 1 / (1 + e^-x)
     */
    public static double sigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    /**
     * dSigmoid()
     * 
     * Derivative of regular sigmoid.
     * 
     * @param x
     * @return sigmoid(x) * (1 - sigmoid(x))
     */
    public static double dSigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

}
