import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    
    public static void main(String[] args) throws Exception {

        String trainingDataFile = "data/Input/train-images.idx3-ubyte";
        String trainingLabelFile = "data/Input/train-labels.idx1-ubyte";
        String testingDataFile = "data/Input/t10k-images.idx3-ubyte";
        String testingLabelFile = "data/Input/t10k-labels.idx1-ubyte";
        MnistDatabase trainingData = new MnistDatabase(trainingDataFile, trainingLabelFile);
        MnistDatabase testingData = new MnistDatabase(testingDataFile, testingLabelFile);
        int[] layerSizes = {50, 10};

        // Population population = new Population(5, layerSizes, trainingData, testingData);

        // population.run();

        // DataPoint ex = trainingData.getDataPoint(0);
        // System.out.println(Arrays.toString(ex.values));

        Network network = new Network(
            "",
            784,
            layerSizes
        );

        double learnRate = 0.05;
        int epochs = 1000000;
        
        int miniBatchSize = 240;

        ArrayList<Integer> scores = new ArrayList<Integer>();

        DataPoint[][] miniBatches = trainingData.generateMiniBatches(miniBatchSize);

        for(int e = 0; e < epochs; e++) {

            int score = 0;

            for(int m = 0; m < miniBatches.length; m++) {

                DataPoint[] miniBatch = miniBatches[m];

                double[][] results = network.learn(miniBatch, learnRate);

                for(int i = 0; i < results.length; i++) {

                    double[] result = results[i];

                    int d = 0;
                    for(int r = 1; r < result.length; r++) {
                        if(result[r] > result[d]) {
                            d = r;
                        }
                    }

                    if(d == miniBatch[i].label) {
                        score++;
                    }

                }
                 
            }

            scores.add(score);
            if(scores.size() > 20) {
                scores.remove(0);
            }

            int avg = 0;
            for(int i = 0; i < scores.size(); i++) {
                avg += scores.get(i);
            }
            avg = (int)(avg / scores.size());

            System.out.println(e + "\t\t" + score + " / " + miniBatchSize * miniBatches.length + " with an average score of " + avg + " / " + miniBatchSize * miniBatches.length);

            if(Math.random() > 0.9) {
                miniBatches = trainingData.generateMiniBatches(miniBatchSize);
                System.out.println("\t\tShuffled the mini-batches.  ");
            }

        }

    }

}
