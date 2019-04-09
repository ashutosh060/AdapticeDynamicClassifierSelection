package DCS.models;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

public class SingleClassifierResult {

    private Classifier classifier;
    private Evaluation trainingSetEvaluation;
    private Evaluation testSetEvaluation;

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    public Evaluation getTrainingSetEvaluation() {
        return trainingSetEvaluation;
    }

    public void setTrainingSetEvaluation(Evaluation trainingSetEvaluation) {
        this.trainingSetEvaluation = trainingSetEvaluation;
    }

    public Evaluation getTestSetEvaluation() {
        return testSetEvaluation;
    }

    public void setTestSetEvaluation(Evaluation testSetEvaluation) {
        this.testSetEvaluation = testSetEvaluation;
    }

}
