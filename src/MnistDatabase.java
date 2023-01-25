import java.io.IOException;

public class MnistDatabase {
    
    private static MnistDatabase instance;
    private DataPoint[] data;

    private MnistDatabase() {}

    public static MnistDatabase getInstance(String dataFile, String labelFile) {
        
        if(instance == null) {
            instance = new MnistDatabase();
            instance.loadData(dataFile, labelFile);
        }

        return instance;
        
    }

    public MnistDatabase loadData(String dataFile, String labelFile) {

        try {
            data = new MnistDataReader().readData(dataFile, labelFile);
        } catch(IOException e) {
            e.printStackTrace();
            return null;
        }

        return new MnistDatabase();

    }

    public DataPoint[] getData() {
        return data;
    }

    public DataPoint getDataPoint(int index) {
        return data[index];
    }

    public DataPoint[] getMiniBatch(int size) {

        DataPoint[] miniBatch = new DataPoint[size];
        boolean[] dupes = new boolean[data.length];
        
        for(int d = 0; d < dupes.length; d++) {
            dupes[d] = false;
        }

        for(int s = 0; s < size; s++) {

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
