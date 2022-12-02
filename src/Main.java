//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Main
 * Driver Class
 * 
 * @author Joshua Savoie
 */
public class Main {

    /**
     * main()
     * 
     * Driver method.
     * 
     * @param args
     */
    public static void main(String[] args) {

        int[] layerSizes = {
            100, 60, 50, 40, 10
        };
        
        Population population = new Population(
            "C:\\Users\\jmps2\\Documents\\dev-Development\\MachineLearning\\NeuralNetwork\\data", 
            20, 
            layerSizes, 
            784, 
            Layer.Functions.SIGMOID
        );
        // Population population = new Population("C:\\Users\\jmps2\\Documents\\dev-Development\\MachineLearning\\NeuralNetwork\\data", 0);

        Simulation simulation = new Simulation(5, population, MnistProcess.class);

        simulation.run();

    }
    
}
