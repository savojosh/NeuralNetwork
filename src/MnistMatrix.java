/**
 * @author turkdogan, Joshua Savoie
 */
public class MnistMatrix {

    private double[][] data;

    private int nRows;
    private int nCols;

    private int label;
    private double[] expectedOutputs;

    public MnistMatrix(int nRows, int nCols) {
        this.nRows = nRows;
        this.nCols = nCols;

        data = new double[nRows][nCols];
    }

    public double getValue(int r, int c) {
        return data[r][c];
    }

    public void setValue(int row, int col, double value) {
        data[row][col] = value;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;

        expectedOutputs = new double[10];
        for(int i = 0; i < 10; i++) {
            expectedOutputs[i] = 0;
        }
        expectedOutputs[label] = 1;
    }

    public double[] getExpectedOutputs() {
        return expectedOutputs;
    }

    public int getNumberOfRows() {
        return nRows;
    }

    public int getNumberOfColumns() {
        return nCols;
    }

}
