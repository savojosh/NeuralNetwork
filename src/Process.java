//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Process
 * Abstract Class
 * 
 * Extend this class to solve a problem.
 * 
 * @author Joshua Savoie
 */
public abstract class Process {

    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private final Network m_network;

    //-----[CONSTRUCTOR]-----\\
    /**
     * Process()
     * 
     * All constructors of sub-classes should have only the
     * Network parameter.
     * 
     * @param network
     */
    public Process(Network network) {

        m_network = network;

    }

    //-----[METHODS]-----\\

    /**
     * output()
     * 
     * Implement this method to complete a given task.
     * 
     * The return should be the cost of the network.
     * 
     * @return
     */
    public abstract double output();

    // TODO: Transition this into a constructor parameter.
    /**
     * setID()
     * 
     * @param id
     */
    public abstract void setID(int id);
    
}
