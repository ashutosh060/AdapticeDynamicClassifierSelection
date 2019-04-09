package DCS;

import weka.core.Instances;

public class TrainingTestSet {

    private Instances trainingSet;
    private Instances testSet;

    public Instances getTrainingSet() {
        return trainingSet;
    }

    public void setTrainingSet(Instances trainingSet) {
        this.trainingSet = trainingSet;
    }

    public Instances getTestSet() {
        return testSet;
    }

    public void setTestSet(Instances testSet) {
        this.testSet = testSet;
    }

}
