//-----[IMPORTS]-----\\

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

//-----[CLASS]-----\\
/**
 * SimulationThread
 * Thread Class
 * 
 * Runs a neural network against a task in its own thread.
 * 
 * @author Joshua Savoie
 */
public class SimulationThread extends Thread {
    
    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private Network m_network;

    private Object m_task;

    private double m_cost;
    private int m_id;

    //-----[CONSTRUCTOR]-----\\
    /**
     * SimulationThread
     * Constructor
     * 
     * Creates a new thread for the given network and task.
     * 
     * Java Reflections is used here.
     * 
     * @param network
     * @param task
     */
    public SimulationThread(Network network, Class<? extends Process> task) {

        m_network = network;

        try {

            // Creates a new instance of the task.
            Constructor<?> constructor = task.getConstructor(Network.class);
            m_task = constructor.newInstance(m_network);

        } catch(Exception e) {

            e.printStackTrace();
            System.exit(0);

        }

        m_cost = 0;

    }

    //-----[METHODS]-----\\

    /**
     * run()
     * 
     * Thread executable.
     */
    public void run() {

        try {

            // Sets the ID.
            Method setID = m_task.getClass().getDeclaredMethod("setID", int.class);
            setID.invoke(m_task, m_id);

            // Runs the task.
            Method output = m_task.getClass().getDeclaredMethod("output");
            m_cost = (double)output.invoke(m_task);

        } catch(Exception e) {

            e.printStackTrace();
            System.exit(0);

        }
    }

    /**
     * getCost()
     * 
     * @return
     */
    public double getCost() {
        return m_cost;
    }

    // TODO: Transition this into a constructor parameter.
    /**
     * setID()
     * 
     * @param id
     */
    public void setID(int id) {
        m_id = id;
    }

}
