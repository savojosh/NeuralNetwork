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
            40, 20, 20, 10
        };

        // Thread t1 = new Thread(new MnistTrainer(
        //     "C:\\Users\\jmps2\\Documents\\dev-Development\\MachineLearning\\NeuralNetwork\\data", 
        //     1, 
        //     784, 
        //     layerSizes, 
        //     Layer.Functions.SIGMOID
        // ));
        
        Population population = new Population(
            "C:\\Users\\jmps2\\Documents\\dev-Development\\MachineLearning\\NeuralNetwork\\data", 
            20, 
            layerSizes, 
            784, 
            Layer.Functions.SIGMOID
        );
        // Population population = new Population("C:\\Users\\jmps2\\Documents\\dev-Development\\MachineLearning\\NeuralNetwork\\data", 43);

        Simulation simulation = new Simulation(5, population, MnistProcess.class);

        simulation.run();

    }
    
}
