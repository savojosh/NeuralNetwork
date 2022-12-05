//-----[IMPORTS]-----\\

import java.util.Scanner;

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

        Scanner keyboard = new Scanner(System.in);

        Population population = null;
        Simulation simulation = null;

        int generation = 0;
        int populationSize = 0;
        int graduationSize = 0;
        int[] layerSizes = null;
        Layer.Functions function = null;
        String storePath = "";

        int populationOption = 0;
        while(populationOption != 1 && populationOption != 2) {
            System.out.println("\nWhat population would you like to use?");
            System.out.println("    1 | New Population");
            System.out.println("    2 | Existing Population");
            populationOption = Integer.parseInt(keyboard.nextLine());
        }

        if(populationOption == 1) {

            System.out.println("\nPopulation size?");
            populationSize = Integer.parseInt(keyboard.nextLine());

            System.out.println("\nLayer sizes?");
            System.out.println("Specify layer sizes by integers split by commas.");
            System.out.println("Example = 16,16,10");
            String[] inputs = keyboard.nextLine().split(",");
            layerSizes = new int[inputs.length];
            for(int l = 0; l < layerSizes.length; l++) layerSizes[l] = Integer.parseInt(inputs[l]);
            
            System.out.println("\nChoose an activation function:");
            for(int f = 0; f < Layer.Functions.values().length; f++) {
                System.out.println("   " + Layer.Functions.values()[f].saveString);
            }
            System.out.println("Copy and paste the name of the desired activation function.");
            while(function == null) {
                String funcName = keyboard.nextLine();
                for(int f = 0; f < Layer.Functions.values().length; f++) {
                    if(funcName.equals(Layer.Functions.values()[f].saveString)) {
                        function = Layer.Functions.values()[f];
                    }
                }
            }

            System.out.println("\nPlease specify an absolute path to a folder to store neural network generations at:");
            storePath = keyboard.nextLine();

        } else if(populationOption == 2) {

            System.out.println("\nWhich generation would you like to start from?");
            generation = Integer.parseInt(keyboard.nextLine());

            System.out.println("\nPlease specify the absolute path to the parent folder of the desired generation.");
            storePath = keyboard.nextLine();

        }

        System.out.println("\nPopulation graduation size?");
        graduationSize = Integer.parseInt(keyboard.nextLine());

        if(populationOption == 1) {
            population = new Population(storePath, populationSize, layerSizes, 784, function);
        } else if(populationOption == 2) {
            population = new Population(storePath, generation);
        }

        simulation = new Simulation(graduationSize, population, MnistProcess.class);

        System.out.println();
        keyboard.close();

        simulation.run();

    }
    
}
