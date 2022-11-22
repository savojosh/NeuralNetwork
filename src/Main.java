import java.util.Arrays;

//-----[IMPORTS]-----\\

//-----[CLASS]-----\\
/**
 * Main
 * Driver Class
 */
public class Main {

    public static void main(String[] args) {

        int[] layerSizes = {
            80, 60, 60, 40, 20, 10
        };

        Thread t1 = new Thread(new MnistTrainer(
            "C:\\Users\\jmps2\\Documents\\dev-Development\\MachineLearning\\NeuralNetwork\\data", 
            1, 
            784, 
            layerSizes, 
            Layer.Functions.SIGMOID
        ));
        // Thread t2 = new Thread(new MnistTrainer(
        //     "C:\\Users\\jmps2\\Documents\\dev-Development\\MachineLearning\\NeuralNetwork\\data", 
        //     2, 
        //     784, 
        //     layerSizes, 
        //     Layer.Functions.SIGMOID
        // ));
        // Thread t3 = new Thread(new MnistTrainer(
        //     "C:\\Users\\jmps2\\Documents\\dev-Development\\MachineLearning\\NeuralNetwork\\data", 
        //     3, 
        //     784, 
        //     layerSizes, 
        //     Layer.Functions.SIGMOID
        // ));

        t1.start();
        // t2.start();
        // t3.start();

    }
    
}
