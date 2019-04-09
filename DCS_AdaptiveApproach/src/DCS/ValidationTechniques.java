package DCS;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import weka.core.Instances;

public class ValidationTechniques {

    public List<Instances> leaveOneOut(String testFileName, Collection<String> fileNames) throws IOException {

        List<Instances> dataset = new ArrayList<>();

        DatasetOperations datasetOperations = new DatasetOperations();

        Instances trainingSet = null;

        for (String fileName : fileNames) {
            if (!fileName.equals(testFileName)) {
                if (trainingSet == null) {
                    trainingSet = new Instances(datasetOperations.loadClearInstancesFromCSV(fileName));
                } else {
                    trainingSet.addAll(datasetOperations.loadClearInstancesFromCSV(fileName));
                }
            }
        }

        Instances testSet = datasetOperations.loadClearInstancesFromCSV(testFileName);

        dataset.add(trainingSet);
        dataset.add(testSet);

        return dataset;
    }

    public List<TrainingTestSet> kFoldsValidation(String fileName, int k) throws IOException {

        List<TrainingTestSet> dataset = new ArrayList<>();

        DatasetOperations datasetOperations = new DatasetOperations();

        Instances data = new Instances(datasetOperations.loadClearInstancesFromCSV(fileName));
        
        int classFeatureIndex = -1;
        
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).name().equals("binaryBug")) {
                classFeatureIndex = i;
                break;
            }
        }

        data.setClassIndex(classFeatureIndex);

        Instances randData = new Instances(data);
        randData.stratify(k);

        for (int i = 0; i < k; i++) {
            TrainingTestSet tts = new TrainingTestSet();
            Instances trainingSet = randData.trainCV(k, i);
            Instances testSet = randData.testCV(k, i);
            trainingSet.setClassIndex(-1);
            testSet.setClassIndex(-1);
            tts.setTrainingSet(trainingSet);
            tts.setTestSet(testSet);
            dataset.add(tts);
        }

        return dataset;
    }
}
