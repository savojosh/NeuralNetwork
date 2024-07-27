//-----[IMPORTS]-----\\

import java.io.IOException;

//-----[CLASS]-----\\
/**
 * MnistDatabase
 * Class
 * 
 * Stores all of the MNIST data.
 * 
 * @author Joshua Savoie
 */
public class MnistDatabase implements Database {

    //-----[VARIABLES]-----\\
    
    private DataPoint[] data;

    //-----[CONSTRUCTOR]-----\\
    /**
     * MnistDatabase
     * Constructor
     * 
     * @param dataFile
     * @param labelFile
     */
    public MnistDatabase(String dataFile, String labelFile) {
        loadData(dataFile, labelFile);
    }

    //-----[METHODS]-----\\
    
    /**
     * loadData()
     * 
     * Loads data from the given files.
     * 
     * @param dataFile - The MNIST file to pull data values from.
     * @param labelFile - The MNIST file to pull correct labels from.
     */
    public void loadData(String dataFile, String labelFile) {

        try {
            data = new MnistDataReader().readData(dataFile, labelFile);
        } catch(IOException e) {
            e.printStackTrace();
        }

    }

    /**
     * getData()
     * 
     * Gets all of the data within this database.
     * 
     * @return
     */
    @Override
    public DataPoint[] getData() {
        return data;
    }

    /**
     * getDataPoint()
     * 
     * Gets the specified data point.
     * 
     * @param index
     * @return
     */
    @Override
    public DataPoint getDataPoint(int index) {
        return data[index];
    }

    /**
     * generateMiniBatches()
     * 
     * Generates some number of minibatche of the data.
     * The mini-batch generated is a set of randomly selected data points with no duplicates.
     * 
     * @param size - The size of the mini-batch.
     * @return
     */
    @Override
    public DataPoint[][] generateMiniBatches(int size) {

        DataPoint[][] miniBatches = new DataPoint[data.length / size][size];

        // Makes sure there is no duplicate data.
        boolean[] dupes = new boolean[data.length];
        for(int d = 0; d < dupes.length; d++) {
            dupes[d] = false;
        }

        for(int m = 0; m < miniBatches.length; m++) {
            for(int s = 0; s < size; s++) {

                // Selects from the database where data.length is excluded. Selection: 0 - (data.length - 1)
                int index = (int)(Math.random() * data.length);
                if(!dupes[index]) {
                    dupes[index] = true;
                    miniBatches[m][s] = data[index]; 
                } else {
                    s--;
                }

            }
        }

        return miniBatches;

    }

}
