//-----[IMPORTS]-----\\

import java.io.IOException;

//-----[CLASS]-----\\
/**
 * MnistDatabase
 * Singleton Class
 * 
 * Stores and provides access to the MNIST images and labels.
 */
public class MnistDatabase {

    //-----[CONSTANTS]-----\\

    //-----[VARIABLES]-----\\

    private static MnistDatabase instance;

    private final MnistMatrix[] training;
    private final MnistMatrix[] testing;

    //-----[CONSTRUCTOR]-----\\
    /**
     * MnistDatabase
     * Singleton Class
     * 
     * Stores and provides access to the MNIST images and labels.
     * 
     * @throws IOException
     */
    private MnistDatabase() throws IOException {
        training = new MnistDataReader().readData("data/Input/train-images.idx3-ubyte", "data/Input/train-labels.idx1-ubyte");
        testing = new MnistDataReader().readData("data/Input/t10k-images.idx3-ubyte", "data/Input/t10k-labels.idx1-ubyte");
    }

    //-----[METHODS]-----\\

    /**
     * getInstance()
     * 
     * Returns the singleton instance of MnistDatabase.
     * @return
     */
    public static MnistDatabase getInstance() {
        if(instance == null) {
            try {
                instance = new MnistDatabase();

                // System.out.println("MnistDatabase created.");
            } catch(IOException e) {
                System.out.println(e.getMessage());
                System.out.println(e.getStackTrace());
                System.out.println(e.getCause());
            }
        }

        return instance;
    }

    /**
     * getTrainingImageLabel()
     * 
     * Returns the label for an image in the training dataset.
     * 
     * @param imageNum
     * @return
     */
    public int getTrainingImageLabel(int imageNum) {

        assert imageNum < training.length: " imageNum out of bounds.";

        MnistMatrix matrix = training[imageNum];

        return matrix.getLabel();

    }

    /**
     * getTrainingImageMatrix()
     * 
     * Returns the pixel values matrix for an image in the training dataset.
     * 
     * @param imageNum
     * @return
     */
    public int[][] getTrainingImageMatrix(int imageNum) {

        assert imageNum < training.length: " imageNum out of bounds.";

        MnistMatrix matrix = training[imageNum];
        int[][] pixels = new int[matrix.getNumberOfRows()][matrix.getNumberOfColumns()];

        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                pixels[r][c] = matrix.getValue(r, c);
            }
        }

        return pixels;
        
    }

    /**
     * getTrainingDatasetSize()
     * 
     * @return
     */
    public int getTrainingDatasetSize() {
        return training.length;
    }

    /**
     * getTestingImageLabel()
     * 
     * Returns the label for an image in the testing dataset.
     * 
     * @param imageNum
     * @return
     */
    public int getTestingImageLabel(int imageNum) {

        assert imageNum < testing.length: " imageNum out of bounds.";

        MnistMatrix matrix = testing[imageNum];

        return matrix.getLabel();

    }

    /**
     * getTestingImageMatrix()
     * 
     * Returns the pixel values matrix for an image in the testing dataset.
     * 
     * @param imageNum
     * @return
     */
    public int[][] getTestingImageMatrix(int imageNum) {

        assert imageNum < testing.length: " imageNum out of bounds.";

        MnistMatrix matrix = testing[imageNum];
        int[][] pixels = new int[matrix.getNumberOfRows()][matrix.getNumberOfColumns()];

        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                pixels[r][c] = matrix.getValue(r, c);
            }
        }

        return pixels;
        
    }

    /**
     * getTestingDatasetSize()
     * 
     * @return
     */
    public int getTestingDatasetSize() {
        return testing.length;
    }
    
}
