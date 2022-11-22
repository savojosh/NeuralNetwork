//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Process
 * Abstract Class
 * 
 * d e s c r i p t i o n
 * 
 * @author Joshua Savoie
 */
public abstract class Process implements Runnable {

    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private final Network m_network;

    private volatile boolean m_stopped;

    //-----[CONSTRUCTOR]-----\\

    public Process(Network network) {
        m_network = network;

        m_stopped = false;
    }

    //-----[METHODS]-----\\

    /**
     * run()
     * 
     * The method that is run automatically after the start() method is called.
     */
    @Override
    public abstract void run();

    
    
}
