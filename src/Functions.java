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
     * binarySigmoid()
     * 
     * Binary sigmoid.
     * 
     * @param x
     * @return f(x) = 1 / (1 + e^-x)
     */
    public static double binarySigmoid(double x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    }

    /**
     * dBinarySigmoid()
     * 
     * Derivative of binary sigmoid.
     * 
     * @param x
     * @return f'(x) = f(x) * (1 - f(x))
     */
    public static double dBinarySigmoid(double x) {
        return binarySigmoid(x) * (1 - binarySigmoid(x));
    }

    /**
     * bipolarSigmoid()
     * 
     * Binary sigmoid.
     * 
     * @param x
     * @return f(x) = 1 / (1 + e^-x)
     */
    public static double bipolarSigmoid(double x) {
        return -1 + 2 / (1 + Math.pow(Math.E, -x));
    }

    /**
     * dBipolarSigmoid()
     * 
     * Derivative of binary sigmoid.
     * 
     * @param x
     * @return f'(x) = 0.5 * (1 + f(x)) * (1 - f(x))
     */
    public static double dBipolarSigmoid(double x) {
        return 0.5 * (1 + bipolarSigmoid(x)) * (1 - bipolarSigmoid(x));
    }

}
