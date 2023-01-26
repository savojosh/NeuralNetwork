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
    
    private static MnistDatabase instance;
    private DataPoint[] data;

    //-----[CONSTRUCTOR]-----\\
    /**
     * MnistDatabase
     * Constructor
     */
    private MnistDatabase() {}

    /**
     * getInstance()
     * 
     * Creates a singleton instance of the MnistDatabase.
     * 
     * @param dataFile - The MNIST file to pull data values from.
     * @param labelFile - The MNIST file to pull correct labels from.
     * @return
     */
    public static MnistDatabase getInstance(String dataFile, String labelFile) {
        
        if(instance == null) {
            instance = new MnistDatabase();
            instance.loadData(dataFile, labelFile);
        }

        return instance;
        
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
     * getMiniBatch()
     * 
     * Gets a mini-batch of the data.
     * The mini-batch generated is a set of randomly selected data points with no duplicates.
     * 
     * @param size - The size of the mini-batch.
     * @return
     */
    @Override
    public DataPoint[] getMiniBatch(int size) {

        DataPoint[] miniBatch = new DataPoint[size];

        // Makes sure there is no duplicate data.
        boolean[] dupes = new boolean[data.length];
        for(int d = 0; d < dupes.length; d++) {
            dupes[d] = false;
        }

        for(int s = 0; s < size; s++) {

            // Selects from the database where data.length is excluded. Selection: 0 - (data.length - 1)
            int index = (int)(Math.random() * data.length);
            if(!dupes[index]) {
                dupes[index] = true;
                miniBatch[s] = data[index];
            } else {
                s--;
            }

        }

        return miniBatch;

    }

}
