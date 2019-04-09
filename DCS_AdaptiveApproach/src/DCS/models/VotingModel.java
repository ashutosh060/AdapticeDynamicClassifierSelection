package DCS.models;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.Vote;
import weka.core.Instances;

public class VotingModel {

    public static SingleClassifierResult buildAndEvaluate(Instances pTrainingSet, Instances pTestSet) throws Exception {

        int classFeatureIndex = 0;

        for (int i = 0; i < pTrainingSet.numAttributes(); i++) {
            if (pTrainingSet.attribute(i).name().equals("binaryBug")) {
                classFeatureIndex = i;
                break;
            }
        }

        pTrainingSet.setClassIndex(classFeatureIndex);
        pTestSet.setClassIndex(classFeatureIndex);

        Vote classifier = new Vote();
        String optionsString = "-B weka.classifiers.bayes.NaiveBayes "
                + "-B weka.classifiers.functions.Logistic "
                + "-B weka.classifiers.functions.MultilayerPerceptron "
                + "-B weka.classifiers.functions.RBFNetwork "
                + "-B weka.classifiers.rules.DecisionTable "
                + "-B weka.classifiers.trees.J48"
                + " -R MAJ";
        String[] options = weka.core.Utils.splitOptions(optionsString);
        classifier.setOptions(options);
        classifier.buildClassifier(pTrainingSet);

        Evaluation eval = new Evaluation(pTrainingSet);
        eval.evaluateModel(classifier, pTestSet);

        SingleClassifierResult result = new SingleClassifierResult();
        result.setClassifier(classifier);
        result.setTestSetEvaluation(eval);

        return result;
    }
}
