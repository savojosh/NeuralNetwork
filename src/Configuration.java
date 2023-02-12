//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Configuration
 * Class
 * 
 * Contains all of the hyperparameters for training.
 * 
 * @author Joshua Savoie
 */
public class Configuration {

    //-----[VARIABLES]-----\\
    
    public final int epochs;
    public final int miniBatchSize;
    public final double learnRate;

    //-----[CONSTRUCTOR]-----\\
    /**
     * Configuration
     * Constructor
     * 
     * Stores the given hyperparameters.
     * 
     * @param epochs - Number of epochs to be elapsed per training session.
     * @param miniBatchSize - Size of the mini-batches of training data per epoch.
     * @param learnRate - Learning rate of the neural network.
     */
    public Configuration(int epochs, int miniBatchSize, double learnRate) {

        this.epochs = epochs;
        this.miniBatchSize = miniBatchSize;
        this.learnRate = learnRate;

    }

    //-----[METHODS]-----\\

    public Configuration clone() {
        return new Configuration(
            epochs,
            miniBatchSize,
            learnRate
        );
    }

}
