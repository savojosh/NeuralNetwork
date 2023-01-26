//-----[INTERFACE]-----\\
/**
 * Database
 * Interface
 * 
 * Provides a general template to build a database from.
 * 
 * @author Joshua Savoie
 */
public interface Database {
    
    public DataPoint[] getData();
    public DataPoint getDataPoint(int index);
    public DataPoint[] getMiniBatch(int size);

}
