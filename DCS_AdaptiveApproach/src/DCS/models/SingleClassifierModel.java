package DCS.models;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class SingleClassifierModel {

    public static SingleClassifierResult buildAndEvaluate(String pClassifierName, Instances pTrainingSet, Instances pTestSet) throws Exception {
        Classifier classifier = null;

        switch (pClassifierName) {
            case "Log":
                classifier = new Logistic();
                break;
            case "DTree":
                classifier = new J48();
                break;
            case "MLP":
                classifier = new MultilayerPerceptron();
                break;
            case "RBF":
                classifier = new RBFNetwork();
                break;
            case "NB":
                classifier = new NaiveBayes();
                break;
            default:
                System.err.println("Unknown classifier.");
        }

        int classFeatureIndex = 0;

        for (int i = 0; i < pTrainingSet.numAttributes(); i++) {
            if (pTrainingSet.attribute(i).name().equals("binaryBug")) {
                classFeatureIndex = i;
                break;
            }
        }

        pTrainingSet.setClassIndex(classFeatureIndex);
        pTestSet.setClassIndex(classFeatureIndex);

        if (classifier != null) {
            classifier.buildClassifier(pTrainingSet);
        }

        Evaluation trainingSetEvaluation = new Evaluation(pTrainingSet);
        trainingSetEvaluation.evaluateModel(classifier, pTrainingSet);
        
        SingleClassifierResult result = new SingleClassifierResult();
        result.setClassifier(classifier);
        result.setTrainingSetEvaluation(trainingSetEvaluation);
        
        Evaluation testSetEvaluation = new Evaluation(pTrainingSet);
        testSetEvaluation.evaluateModel(classifier, pTestSet);
        
        result.setClassifier(classifier);
        result.setTestSetEvaluation(testSetEvaluation);
        
        return result;
    }
}
