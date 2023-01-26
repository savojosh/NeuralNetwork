import java.util.ArrayList;

public class Main {
    
    public static void main(String[] args) throws Exception {

        String dataFile = "data/Input/train-images.idx3-ubyte";
        String labelFile = "data/Input/train-labels.idx1-ubyte";
        MnistDatabase database = MnistDatabase.getInstance(dataFile, labelFile);
        int[] layerSizes = {16, 16, 10};
        Network network = new Network(
            "",
            784,
            layerSizes
        );

        double learnRate = 0.1;
        int epochs = 1000000;
        int miniBatchSize = 2000;
        ArrayList<Integer> scores = new ArrayList<Integer>();

        DataPoint[] miniBatch = database.getMiniBatch(miniBatchSize);

        for(int e = 0; e < epochs; e++) {

            miniBatch = database.getMiniBatch(miniBatchSize);

            double[][] results = network.learn(miniBatch, learnRate);

            int score = 0;
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

            scores.add(score);
            if(scores.size() > 20) {
                scores.remove(0);
            }

            int avg = 0;
            for(int i = 0; i < scores.size(); i++) {
                avg += scores.get(i);
            }
            avg = (int)(avg / scores.size());

            System.out.println(e + "\t\t" + score + " / " + miniBatch.length + " with an average score of " + avg + " / " + miniBatch.length);

        }

    }

}