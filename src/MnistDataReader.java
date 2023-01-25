import java.io.*;
/**
 * @author turkdogan, Joshua Savoie
 */
public class MnistDataReader  {

    public DataPoint[] readData(String dataFilePath, String labelFilePath) throws IOException {

        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(dataFilePath)));
        int magicNumber = dataInputStream.readInt();
        int numberOfItems = dataInputStream.readInt();
        int nRows = dataInputStream.readInt();
        int nCols = dataInputStream.readInt();

        // System.out.println("magic number is " + magicNumber);
        // System.out.println("number of items is " + numberOfItems);
        // System.out.println("number of rows is: " + nRows);
        // System.out.println("number of cols is: " + nCols);

        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelFilePath)));
        int labelMagicNumber = labelInputStream.readInt();
        int numberOfLabels = labelInputStream.readInt();

        // System.out.println("labels magic number is: " + labelMagicNumber);
        // System.out.println("number of labels is: " + numberOfLabels);

        DataPoint[] data = new DataPoint[numberOfItems];

        assert numberOfItems == numberOfLabels;

        for(int i = 0; i < numberOfItems; i++) {

            int label = labelInputStream.readUnsignedByte();
            double[] values = new double[nRows * nCols];
            double[] y = new double[10];
            
            int v = 0;
            for(int r = 0; r < nRows; r++) {
                for(int c = 0; c < nCols; c++) {
                    values[v] = (double)(dataInputStream.readUnsignedByte());
                    v++;
                }
            }

            for(int e = 0; e < y.length; e++) {
                y[e] = 0.0;
            }
            y[label] = 1.0;
            
            data[i] = new DataPoint(label, values, y);

        }

        dataInputStream.close();
        labelInputStream.close();

        return data;
    }
}
