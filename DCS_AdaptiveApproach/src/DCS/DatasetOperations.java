package DCS;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class DatasetOperations {

    private static final String COMMA_DELIMITER = ",";
    private static final String NEW_LINE_SEPARATOR = "\n";

    public Instances loadInstancesFromCSV(String pFileName) {
        Instances instances = null;
        CSVLoader loader = new CSVLoader();
        try {
            loader.setSource(new File(pFileName));
            instances = loader.getDataSet();
        } catch (IOException ex) {
            Logger.getLogger(DatasetOperations.class.getName()).log(Level.SEVERE, null, ex);
        }
        return instances;
    }

    public Instances loadClearInstancesFromCSV(String pFileName) {
        Instances instances = this.loadInstancesFromCSV(pFileName);

        //Convert bugs from integer to binary
        List binaryBugValues = new ArrayList();
        binaryBugValues.add("TRUE");
        binaryBugValues.add("FALSE");
        instances.insertAttributeAt(new Attribute("binaryBug", binaryBugValues), instances.numAttributes());

        for (Instance instance : instances) {
            if (instance.value(instances.numAttributes() - 2) > 0) {
                instance.setValue(instances.numAttributes() - 1, "TRUE");
            } else {
                instance.setValue(instances.numAttributes() - 1, "FALSE");
            }
        }
        instances.deleteAttributeAt(instances.numAttributes() - 2);

        return instances;
    }

    public void makeCSV(String pNewFileName, Instances pInstances, boolean pEmptyCSV) throws IOException {
        try (FileWriter fileWriter = new FileWriter(pNewFileName)) {
            for (int column = 0; column < pInstances.numAttributes(); column++) {
                fileWriter.append(pInstances.attribute(column).name());
                if (column != pInstances.numAttributes() - 1) {
                    fileWriter.append(COMMA_DELIMITER);
                }
            }
            if (!pEmptyCSV) {
                for (int row = 0; row < pInstances.numInstances(); row++) {
                    fileWriter.append(NEW_LINE_SEPARATOR);
                    double[] temp = pInstances.instance(row).toDoubleArray();
                    for (int column = 0; column < pInstances.numAttributes(); column++) {
                        fileWriter.append(String.valueOf(temp[column]));
                        if (column != pInstances.numAttributes() - 1) {
                            fileWriter.append(COMMA_DELIMITER);
                        }
                    }
                }
            }
            fileWriter.flush();
        }
    }

    public void fromClustersToCSV(String pFileName, Instance pInstance) throws IOException {
        try (FileWriter fileWriter = new FileWriter(pFileName, true)) {
            double[] temp = null;
            fileWriter.append(NEW_LINE_SEPARATOR);
            for (int column = 0; column < pInstance.numAttributes() - 1; column++) {
                temp = pInstance.toDoubleArray();
                fileWriter.append(String.valueOf(temp[column]));
                fileWriter.append(COMMA_DELIMITER);
            }
            if (temp[pInstance.numAttributes() - 1] == 0) {
                fileWriter.append("FALSE");
            } else {
                fileWriter.append("TRUE");
            }
            fileWriter.flush();
        }
    }

    public void addInstancesToCSV(String pFileName, Instances pInstances) throws IOException {
        try (FileWriter fileWriter = new FileWriter(pFileName, true)) {
            for (int row = 0; row < pInstances.numInstances(); row++) {
                fileWriter.append(NEW_LINE_SEPARATOR);
                double[] temp = pInstances.instance(row).toDoubleArray();
                for (int column = 0; column < pInstances.numAttributes(); column++) {
                    fileWriter.append(String.valueOf(temp[column]));
                    if (column != pInstances.numAttributes() - 1) {
                        fileWriter.append(COMMA_DELIMITER);
                    }
                }
            }
            fileWriter.flush();
        }
    }

}
